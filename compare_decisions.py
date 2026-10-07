"""
compare_decisions.py

Benchmark local decision / classification models side-by-side on one or more
datasets of choice questions. Everything runs locally: weights are downloaded
from the Hugging Face Hub on first run and cached, then inference runs on CPU
(or GPU if PyTorch sees one). No paid APIs, no background services.

Usage:
    python compare_decisions.py
    python compare_decisions.py --models local_onnx_deberta local_gliclass
    python compare_decisions.py --dataset decision_choices path/to/other.json
    python compare_decisions.py --dataset default      # the single built-in test case
    python compare_decisions.py --runs 5
    python compare_decisions.py --prepare             # download + test-load models only

Models are listed in models.json (key, name, type, Hugging Face model_id).
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import io
import json
import logging
import os
import statistics
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv

# Load optional settings (HF_TOKEN, HF_HOME, etc.) before importing HF libraries
# so that they pick up cache locations from the environment.
load_dotenv()

import pandas as pd  # noqa: E402
import torch  # noqa: E402

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Toggle models here: comment out any entry to skip it in the benchmark run.
# Keys refer to entries in models.json.
TARGET_MODELS: list[str] = [
    "local_onnx_deberta",
    "local_pytorch_deberta",
    "local_bge_reranker",
    "local_gliclass",
]

# Datasets to run by default. Each entry is a path to a JSON file, the name of a
# file in DATASETS_DIR (with or without .json), or "default" for the built-in
# single test case below.
DATASETS: list[str] = [
    "decision_choices",
]
DATASETS_DIR = Path(__file__).resolve().parent / "datasets"

# Model list shared with dashboard.py. Each entry's "type" picks a loader from MODEL_TYPES.
MODELS_FILE = Path(__file__).resolve().parent / "models.json"

# Default benchmark test case (available as the "default" dataset).
STATE = (
    "User message: 'I was double-billed $49.99 for my subscription this month, "
    "please issue a refund immediately.'"
)
CANDIDATE_OPTIONS = [
    "billing_issue",
    "technical_bug",
    "general_question",
    "cancellation_request",
]

# Sentence each option is slotted into for NLI / reranker scoring, unless a
# dataset or item sets its own "hypothesis_template".
DEFAULT_TEMPLATE = "This message is a {}."

# Number of timed inference runs per question (after one untimed warm-up run).
DEFAULT_RUNS = 3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

@dataclass
class Item:
    id: str
    state: str
    options: list[str]
    expected: str | None
    template: str


@dataclass
class Dataset:
    name: str
    items: list[Item]


def resolve_dataset_path(spec: str) -> Path:
    for candidate in (Path(spec), DATASETS_DIR / spec, DATASETS_DIR / f"{spec}.json"):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"dataset not found: {spec!r} (also looked in {DATASETS_DIR})")


def load_dataset(spec: str) -> Dataset:
    """Load a dataset from JSON: either {"name", "hypothesis_template", "items": [...]} or a bare list of items."""
    if spec == "default":
        item = Item("double_billing", STATE, CANDIDATE_OPTIONS, "billing_issue", DEFAULT_TEMPLATE)
        return Dataset("default", [item])

    path = resolve_dataset_path(spec)
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        data = {"items": data}
    default_template = data.get("hypothesis_template", DEFAULT_TEMPLATE)

    items = []
    for i, raw in enumerate(data.get("items", []), 1):
        item = Item(
            id=str(raw.get("id", i)),
            state=raw["state"],
            options=list(raw["options"]),
            expected=raw.get("expected"),
            template=raw.get("hypothesis_template", default_template),
        )
        problems = []
        if len(item.options) < 2:
            problems.append("needs at least 2 options")
        if item.expected is not None and item.expected not in item.options:
            problems.append(f"expected {item.expected!r} is not one of the options")
        if "{}" not in item.template:
            problems.append("hypothesis_template must contain {}")
        if problems:
            raise ValueError(f"{path.name}, item {item.id}: {'; '.join(problems)}")
        items.append(item)

    if not items:
        raise ValueError(f"{path.name}: no items")
    return Dataset(data.get("name", path.stem), items)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class Prediction:
    choice: str
    score: float


# A loader returns a predict function: (state, options, template) -> Prediction.
Predictor = Callable[[str, list[str], str], Prediction]


def humanize(label: str) -> str:
    """Turn 'billing_issue' into 'billing issue' so NLI/reranker models read natural text."""
    return label.replace("_", " ")


def zero_shot_predictor(classifier) -> Predictor:
    """Wrap a transformers zero-shot-classification pipeline."""

    def predict(state: str, options: list[str], template: str) -> Prediction:
        readable = [humanize(o) for o in options]
        result = classifier(
            state,
            candidate_labels=readable,
            hypothesis_template=template,
            multi_label=False,
        )
        top = readable.index(result["labels"][0])
        return Prediction(options[top], float(result["scores"][0]))

    return predict


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

def load_zero_shot_onnx(model_id: str) -> Predictor:
    from optimum.onnxruntime import ORTModelForSequenceClassification
    from transformers import AutoTokenizer, pipeline

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # export=True converts the PyTorch checkpoint to ONNX on load.
    model = ORTModelForSequenceClassification.from_pretrained(model_id, export=True)
    classifier = pipeline("zero-shot-classification", model=model, tokenizer=tokenizer)
    return zero_shot_predictor(classifier)


def load_zero_shot_pytorch(model_id: str) -> Predictor:
    from transformers import pipeline

    classifier = pipeline(
        "zero-shot-classification",
        model=model_id,
        device=0 if DEVICE == "cuda" else -1,
    )
    return zero_shot_predictor(classifier)


def load_reranker(model_id: str) -> Predictor:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model.to(DEVICE).eval()

    def predict(state: str, options: list[str], template: str) -> Prediction:
        # Score each (state, candidate) pair; the highest relevance wins.
        pairs = [[state, template.format(humanize(o))] for o in options]
        inputs = tokenizer(
            pairs, padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to(DEVICE)
        with torch.inference_mode():
            logits = model(**inputs).logits.float()
        # Most rerankers output one relevance logit per pair; two-class ones use the "relevant" column.
        relevance = logits[:, -1]
        # Softmax across candidates gives a 0-1 confidence comparable to the other models.
        probs = torch.softmax(relevance, dim=0)
        top = int(torch.argmax(probs))
        return Prediction(options[top], float(probs[top]))

    return predict


def load_gliclass(model_id: str) -> Predictor:
    from gliclass import GLiClassModel, ZeroShotClassificationPipeline
    from transformers import AutoTokenizer

    model = GLiClassModel.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    classifier = ZeroShotClassificationPipeline(
        model,
        tokenizer,
        classification_type="multi-label",
        device="cuda:0" if DEVICE == "cuda" else "cpu",
    )

    def predict(state: str, options: list[str], template: str) -> Prediction:
        # GLiClass scores label names directly, so the hypothesis template is not used.
        readable = [humanize(o) for o in options]
        # The pipeline prints a tqdm progress bar on every call; keep it out of the output.
        with contextlib.redirect_stderr(io.StringIO()):
            # threshold=0.0 returns a score for every label so we can always pick a top choice.
            results = classifier(state, readable, threshold=0.0)[0]
        best = max(results, key=lambda r: r["score"])
        return Prediction(options[readable.index(best["label"])], float(best["score"]))

    return predict


# Model "type" values allowed in models.json.
MODEL_TYPES: dict[str, Callable[[str], Predictor]] = {
    "zero_shot_onnx": load_zero_shot_onnx,
    "zero_shot_pytorch": load_zero_shot_pytorch,
    "reranker": load_reranker,
    "gliclass": load_gliclass,
}


def load_model_registry() -> dict[str, tuple[str, Callable[[], Predictor]]]:
    """Read models.json into {key: (display name, loader)}."""
    try:
        entries = json.loads(MODELS_FILE.read_text(encoding="utf-8"))["models"]
    except (OSError, ValueError, KeyError) as exc:
        sys.exit(f"Could not read {MODELS_FILE.name}: {exc}")
    registry = {}
    for m in entries:
        loader = MODEL_TYPES.get(m.get("type"))
        if loader is None:
            print(f"Skipping model {m.get('key')!r}: unknown type {m.get('type')!r} in {MODELS_FILE.name}")
            continue
        registry[m["key"]] = (m.get("name", m["key"]), functools.partial(loader, m["model_id"]))
    return registry


MODEL_REGISTRY = load_model_registry()


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def benchmark_model(key: str, datasets: list[Dataset], runs: int) -> tuple[list[dict], list[dict]]:
    """Load one model and run it over every dataset. Returns (summary rows, per-item rows)."""
    name, loader = MODEL_REGISTRY[key]
    print(f"\n==> {name} [{key}]")

    try:
        print("    loading (first run downloads weights)...")
        t0 = time.perf_counter()
        predict = loader()
        load_s = time.perf_counter() - t0
        warmup = datasets[0].items[0]
        predict(warmup.state, warmup.options, warmup.template)  # warm-up, not timed
    except Exception as exc:  # keep benchmarking the remaining models
        print(f"    !! failed: {type(exc).__name__}: {exc}")
        status = f"error: {type(exc).__name__}"
        return [{"Dataset": ds.name, "Model": name, "Status": status} for ds in datasets], []

    summary, details = [], []
    for ds in datasets:
        timings_ms, scores, correct, graded, errors = [], [], 0, 0, 0
        for n, item in enumerate(ds.items, 1):
            progress = f"    [{n}/{len(ds.items)}] {item.id}"
            item_ms = []
            try:
                for _ in range(runs):
                    t0 = time.perf_counter()
                    prediction = predict(item.state, item.options, item.template)
                    item_ms.append((time.perf_counter() - t0) * 1000)
            except Exception as exc:
                errors += 1
                print(f"{progress}: ERROR {type(exc).__name__}: {exc}", flush=True)
                details.append({
                    "Dataset": ds.name,
                    "Item": item.id,
                    "Expected": item.expected,
                    "Model": key,
                    "Choice": "ERROR",
                    "Correct": None,
                })
                continue

            timings_ms.extend(item_ms)
            is_correct = None if item.expected is None else prediction.choice == item.expected
            if is_correct is not None:
                graded += 1
                correct += is_correct
            scores.append(prediction.score)
            mark = "" if is_correct is None else (" (ok)" if is_correct else " (WRONG)")
            print(f"{progress} -> {prediction.choice}{mark}, {statistics.median(item_ms):.0f} ms", flush=True)
            details.append({
                "Dataset": ds.name,
                "Item": item.id,
                "Expected": item.expected,
                "Model": key,
                "Choice": prediction.choice,
                "Score": prediction.score,
                "Correct": is_correct,
                "Latency (ms)": round(statistics.median(item_ms), 1),
            })

        latency = statistics.median(timings_ms) if timings_ms else None
        accuracy = correct / graded if graded else None
        print(
            f"    {ds.name}: {correct}/{graded} correct"
            + (f", median {latency:.1f} ms/question" if latency is not None else "")
        )
        summary.append({
            "Dataset": ds.name,
            "Model": name,
            "Accuracy": round(accuracy, 3) if accuracy is not None else None,
            "Correct": f"{correct}/{graded}",
            "Avg Confidence": round(statistics.mean(scores), 4) if scores else None,
            "Latency (ms)": round(latency, 1) if latency is not None else None,
            "Min Latency (ms)": round(min(timings_ms), 1) if timings_ms else None,
            "Load Time (s)": round(load_s, 1),
            "Status": "ok" if not errors else f"{errors} item error(s)",
        })
    return summary, details


def print_dataset_report(ds: Dataset, summary: list[dict], details: list[dict], model_keys: list[str]) -> None:
    print("\n" + "=" * 100)
    print(f"DATASET: {ds.name} ({len(ds.items)} questions)")
    print("=" * 100)

    # Per-question choices, one column per model; wrong answers are marked with *.
    lookup = {(d["Item"], d["Model"]): d for d in details if d["Dataset"] == ds.name}
    rows = []
    for item in ds.items:
        row = {"Question": item.id, "Expected": item.expected or "-"}
        for key in model_keys:
            d = lookup.get((item.id, key))
            row[key.removeprefix("local_")] = (
                "-" if d is None else d["Choice"] + (" *" if d["Correct"] is False else "")
            )
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))
    print("(* = wrong answer)\n")

    df = pd.DataFrame([s for s in summary if s["Dataset"] == ds.name]).drop(columns="Dataset")
    if "Accuracy" in df:
        df = df.sort_values(["Accuracy", "Latency (ms)"], ascending=[False, True], na_position="last")
    print(df.to_string(index=False))


def prepare_models(keys: list[str]) -> int:
    """Download (if needed) and test-load each model with one prediction. Returns an exit code."""
    failed = 0
    for key in keys:
        name, loader = MODEL_REGISTRY[key]
        print(f"\n==> {name} [{key}]: downloading if needed, then loading...", flush=True)
        try:
            t0 = time.perf_counter()
            predict = loader()
            prediction = predict(STATE, CANDIDATE_OPTIONS, DEFAULT_TEMPLATE)
            print(f"    OK in {time.perf_counter() - t0:.1f} s (test answer: {prediction.choice})", flush=True)
        except Exception as exc:
            failed += 1
            print(f"    FAILED: {type(exc).__name__}: {exc}", flush=True)
    print(f"\n{len(keys) - failed}/{len(keys)} models ready.")
    return 1 if failed else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark local decision models side-by-side.")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_REGISTRY),
        default=[key for key in TARGET_MODELS if key in MODEL_REGISTRY],
        help="Model keys from models.json (defaults to TARGET_MODELS).",
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        dest="datasets",
        default=DATASETS,
        help="Dataset JSON paths, names in datasets/, or 'default' (defaults to DATASETS).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=DEFAULT_RUNS,
        help=f"Timed inference runs per question after warm-up (default {DEFAULT_RUNS}).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help="CPU threads for the PyTorch models (default 0 = PyTorch's own choice).",
    )
    parser.add_argument(
        "--results-json",
        type=Path,
        help="Also save the results to this JSON file (used by dashboard.py).",
    )
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Only download and test-load the selected models, then exit.",
    )
    args = parser.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    if args.prepare:
        sys.exit(prepare_models(args.models))

    # Load datasets before any model so a bad file fails fast.
    try:
        datasets = [load_dataset(spec) for spec in args.datasets]
    except (OSError, ValueError, KeyError) as exc:
        parser.error(f"could not load dataset: {exc}")

    print(f"Device: {DEVICE} | torch threads: {torch.get_num_threads()} | runs/question: {args.runs}")
    for ds in datasets:
        print(f"Dataset: {ds.name} ({len(ds.items)} questions)")

    summary, details = [], []
    for key in args.models:
        model_summary, model_details = benchmark_model(key, datasets, max(1, args.runs))
        summary += model_summary
        details += model_details

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    print("\nLatency = median per-question inference time, after warm-up.")
    for ds in datasets:
        print_dataset_report(ds, summary, details, args.models)

    if args.results_json:
        payload = {
            "models": [{"key": key, "name": MODEL_REGISTRY[key][0]} for key in args.models],
            "datasets": [
                {"name": ds.name, "items": [{"id": i.id, "expected": i.expected} for i in ds.items]}
                for ds in datasets
            ],
            "summary": summary,
            "details": details,
        }
        args.results_json.parent.mkdir(parents=True, exist_ok=True)
        args.results_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nResults saved to {args.results_json}")


if __name__ == "__main__":
    main()
