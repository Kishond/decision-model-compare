"""
dashboard.py

Desktop dashboard for compare_decisions.py: set up the Python environment,
choose or add models and datasets, set run options, press Start, and read the
results in tables.

The benchmark itself runs as a separate process with the project's .venv
Python, so this file only uses the standard library and can be packaged as a
small .exe (see build_exe.ps1). The exe must stay in the project folder.
"""

from __future__ import annotations

import csv
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import tkinter as tk
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

APP_TITLE = "Decision Model Compare"
BUILTIN_DATASET = "default"

# Keep in sync with MODEL_TYPES in compare_decisions.py: type -> (label, example models).
MODEL_TYPES = {
    "zero_shot_onnx": (
        "Zero-shot NLI, ONNX Runtime (fast on CPU)",
        "NLI / zero-shot models, e.g. facebook/bart-large-mnli,\nMoritzLaurer/deberta-v3-large-zeroshot-v2.0",
    ),
    "zero_shot_pytorch": (
        "Zero-shot NLI, PyTorch",
        "NLI / zero-shot models, e.g. facebook/bart-large-mnli,\nMoritzLaurer/deberta-v3-large-zeroshot-v2.0",
    ),
    "reranker": (
        "Cross-encoder reranker",
        "Rerankers, e.g. BAAI/bge-reranker-v2-m3, BAAI/bge-reranker-base",
    ),
    "gliclass": (
        "GLiClass",
        "GLiClass models, e.g. knowledgator/gliclass-base-v1.0",
    ),
}

SUMMARY_COLUMNS = [
    "Model", "Accuracy", "Correct", "Avg Confidence",
    "Latency (ms)", "Min Latency (ms)", "Load Time (s)", "Status",
]
DETAIL_EXPORT_COLUMNS = [
    "Dataset", "Item", "Expected", "Model", "Choice", "Score", "Correct", "Latency (ms)",
]

# Matches the per-question progress lines printed by compare_decisions.py.
PROGRESS_LINE = re.compile(r"^\s+\[\d+/\d+\]")

# Python versions that torch and onnxruntime ship Windows wheels for, newest first.
SUPPORTED_PYTHON = [(3, 13), (3, 12), (3, 11), (3, 10)]
TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
PYTHON_DOWNLOAD_URL = "https://www.python.org/downloads/windows/"
IMPORT_CHECK = (
    "import importlib.metadata as m\n"
    "import torch, transformers, optimum.onnxruntime, gliclass, pandas, dotenv\n"
    "for p in ['torch', 'transformers', 'optimum', 'onnxruntime', 'gliclass', 'pandas', 'python-dotenv']:\n"
    "    print(f'  {p:14} {m.version(p)}')\n"
    "print('All packages import correctly.')\n"
)
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def find_project_dir() -> Path:
    """Folder with compare_decisions.py: next to this script/exe, or one level up."""
    here = Path(sys.executable if getattr(sys, "frozen", False) else __file__).resolve().parent
    for candidate in (here, here.parent):
        if (candidate / "compare_decisions.py").is_file():
            return candidate
    return here


PROJECT_DIR = find_project_dir()
SCRIPT = PROJECT_DIR / "compare_decisions.py"
REQUIREMENTS = PROJECT_DIR / "requirements.txt"
VENV_DIR = PROJECT_DIR / ".venv"
VENV_PYTHON = VENV_DIR / "Scripts" / "python.exe"
DATASETS_DIR = PROJECT_DIR / "datasets"
MODELS_FILE = PROJECT_DIR / "models.json"
RESULTS_FILE = PROJECT_DIR / "results" / "last_run.json"
SETTINGS_FILE = PROJECT_DIR / "dashboard_settings.json"


# ---------------------------------------------------------------------------
# Helpers that don't need the window
# ---------------------------------------------------------------------------

def check_dataset(path: Path) -> tuple[int, str | None]:
    """Return (question count, error message or None) for a dataset JSON file."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        items = data if isinstance(data, list) else data.get("items", [])
        if not items:
            return 0, "no items"
        for i, item in enumerate(items, 1):
            name = item.get("id", i)
            if "state" not in item or "options" not in item:
                return 0, f"question {name} needs 'state' and 'options'"
            if len(item["options"]) < 2:
                return 0, f"question {name} needs at least 2 options"
            if item.get("expected") is not None and item["expected"] not in item["options"]:
                return 0, f"question {name}: expected answer is not one of the options"
        return len(items), None
    except (OSError, ValueError, AttributeError, TypeError) as exc:
        return 0, str(exc)


def load_models() -> list[dict]:
    try:
        return json.loads(MODELS_FILE.read_text(encoding="utf-8"))["models"]
    except (OSError, ValueError, KeyError):
        return []


def save_models(models: list[dict]) -> None:
    MODELS_FILE.write_text(json.dumps({"models": models}, indent=2) + "\n", encoding="utf-8")


def find_base_python() -> str | None:
    """Find a 64-bit Python 3.10-3.13 on this PC to create .venv with."""
    probe = "import sys, struct; print(sys.executable); print(*sys.version_info[:2], struct.calcsize('P') * 8)"
    candidates = [["py", f"-{major}.{minor}-64"] for major, minor in SUPPORTED_PYTHON]
    candidates += [["python"], ["python3"]]
    for prefix in candidates:
        try:
            out = subprocess.run(
                [*prefix, "-c", probe], capture_output=True, text=True, timeout=30, creationflags=NO_WINDOW
            )
            exe, version = out.stdout.strip().splitlines()[-2:]
            major, minor, bits = map(int, version.split())
        except (OSError, subprocess.TimeoutExpired, ValueError):
            continue
        if out.returncode == 0 and (major, minor) in SUPPORTED_PYTHON and bits == 64:
            return exe
    return None


def hub_model_info(model_id: str) -> tuple[dict | None, str | None]:
    """Look a model up on the Hugging Face Hub. Returns (info, error)."""
    url = "https://huggingface.co/api/models/" + urllib.parse.quote(model_id, safe="/")
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            return json.load(response), None
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 404):
            return None, "not found"
        return None, f"Hugging Face returned HTTP {exc.code}"
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return None, f"could not reach Hugging Face ({exc})"


NLI_MARKERS = {"nli", "mnli", "xnli", "anli", "snli", "zeroshot"}


def guess_model_type(info: dict) -> tuple[str | None, str]:
    """Guess the model type from its Hugging Face info. Returns (type or None, reason)."""
    task = info.get("pipeline_tag")
    tags = [str(t).lower() for t in info.get("tags") or []]
    text = " ".join([str(info.get("id", "")).lower(), str(info.get("library_name") or "").lower(), *tags])
    tokens = set(re.split(r"[^a-z0-9]+", text))

    if "gliclass" in text:
        return "gliclass", "it's a GLiClass model"
    # NLI is checked before rerankers: NLI cross-encoders must run as zero-shot classifiers.
    if task == "zero-shot-classification" or "zero-shot-classification" in tags or tokens & NLI_MARKERS:
        return "zero_shot_onnx", "it's a zero-shot / NLI model"
    if task == "text-ranking" or "rerank" in text or "cross-encoder" in text:
        return "reranker", "it's a reranker"
    if task:
        return None, f"its Hugging Face task is '{task}', which doesn't match a supported type"
    return None, "Hugging Face doesn't say what task it's for"


def same_family(a: str, b: str) -> bool:
    """ONNX and PyTorch zero-shot are the same kind of model, just run differently."""
    return a.removesuffix("_onnx").removesuffix("_pytorch") == b.removesuffix("_onnx").removesuffix("_pytorch")


def clean_model_id(text: str) -> str:
    """Accept 'org/name' or a pasted https://huggingface.co/org/name/... link."""
    text = text.strip()
    text = re.sub(r"^https?://(www\.)?huggingface\.co/", "", text)
    return "/".join(text.strip("/").split("/")[:2])


def fmt(value) -> str:
    return "-" if value is None else str(value)


def elapsed_text(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 60}:{seconds % 60:02d}"


# ---------------------------------------------------------------------------
# Add model dialog
# ---------------------------------------------------------------------------

class AddModelDialog(tk.Toplevel):
    def __init__(self, parent: Dashboard) -> None:
        super().__init__(parent)
        self.parent = parent
        self.title("Add model")
        self.transient(parent)
        self.resizable(False, False)
        # model_id -> (info, error); filled in by background lookups.
        self.hub_cache: dict[str, tuple[dict | None, str | None]] = {}
        self.detect_job: str | None = None

        body = ttk.Frame(self, padding=14)
        body.pack(fill="both", expand=True)
        self.id_var = tk.StringVar()
        self.name_var = tk.StringVar()
        self.hint_var = tk.StringVar()
        self.type_var = tk.StringVar(value=MODEL_TYPES["zero_shot_onnx"][0])

        ttk.Label(body, text="Hugging Face model ID:").grid(row=0, column=0, sticky="w")
        id_entry = ttk.Entry(body, textvariable=self.id_var, width=48)
        id_entry.grid(row=0, column=1, sticky="we", pady=3)
        ttk.Label(body, text="for example  BAAI/bge-reranker-base  (or paste the page link)", foreground="gray").grid(
            row=1, column=1, sticky="w"
        )

        ttk.Label(body, text="Model type:").grid(row=2, column=0, sticky="w", pady=(10, 3))
        type_box = ttk.Combobox(
            body, textvariable=self.type_var, state="readonly", width=45,
            values=[label for label, _ in MODEL_TYPES.values()],
        )
        type_box.grid(row=2, column=1, sticky="we", pady=(10, 3))
        type_box.bind("<<ComboboxSelected>>", lambda _e: self.update_examples())
        self.detected = ttk.Label(body, text="Type a model ID and the type is filled in automatically.",
                                  foreground="gray", wraplength=380, justify="left")
        self.detected.grid(row=3, column=1, sticky="w")
        self.examples = ttk.Label(body, foreground="gray", justify="left")
        self.examples.grid(row=4, column=1, sticky="w", pady=(4, 0))

        ttk.Label(body, text="Display name (optional):").grid(row=5, column=0, sticky="w", pady=(10, 3))
        ttk.Entry(body, textvariable=self.name_var, width=48).grid(row=5, column=1, sticky="we", pady=(10, 3))
        ttk.Label(body, text="Note (optional):").grid(row=6, column=0, sticky="w", pady=3)
        ttk.Entry(body, textvariable=self.hint_var, width=48).grid(row=6, column=1, sticky="we", pady=3)

        buttons = ttk.Frame(body)
        buttons.grid(row=7, column=0, columnspan=2, sticky="e", pady=(12, 0))
        self.add_btn = ttk.Button(buttons, text="Add", command=self.submit)
        self.add_btn.pack(side="left")
        ttk.Button(buttons, text="Cancel", command=self.destroy).pack(side="left", padx=(6, 0))

        self.update_examples()
        self.id_var.trace_add("write", self.schedule_detect)
        self.bind("<Return>", lambda _e: self.submit())
        self.bind("<Escape>", lambda _e: self.destroy())
        id_entry.focus_set()
        self.grab_set()

    def selected_type(self) -> str:
        label = self.type_var.get()
        return next(key for key, (lbl, _) in MODEL_TYPES.items() if lbl == label)

    def update_examples(self) -> None:
        self.examples.configure(text=MODEL_TYPES[self.selected_type()][1])

    def show_detected(self, text: str, color: str = "gray") -> None:
        self.detected.configure(text=text, foreground=color)

    # ----- Hugging Face lookup ---------------------------------------------------

    def lookup(self, model_id: str, then) -> None:
        """Fetch model info in the background (once per ID), then call then(model_id) on the UI thread."""
        if model_id in self.hub_cache:
            then(model_id)
            return

        def fetch() -> None:
            self.hub_cache[model_id] = hub_model_info(model_id)

        threading.Thread(target=fetch, daemon=True).start()

        def wait() -> None:
            if not self.winfo_exists():  # dialog was closed while looking up
                return
            if model_id not in self.hub_cache:
                self.after(100, wait)
                return
            then(model_id)

        self.after(100, wait)

    def schedule_detect(self, *_args) -> None:
        # Wait until typing pauses before looking the model up.
        if self.detect_job:
            self.after_cancel(self.detect_job)
        self.detect_job = self.after(700, self.start_detect)

    def start_detect(self) -> None:
        self.detect_job = None
        model_id = clean_model_id(self.id_var.get())
        if not re.fullmatch(r"[\w.-]+/[\w.-]+", model_id):
            self.show_detected("Type a model ID and the type is filled in automatically.")
            return
        self.show_detected("Looking up the model on Hugging Face...")
        self.lookup(model_id, self.apply_detection)

    def apply_detection(self, model_id: str) -> None:
        if model_id != clean_model_id(self.id_var.get()):
            return  # the ID changed while we were looking it up
        info, error = self.hub_cache[model_id]
        if error == "not found":
            self.show_detected("Not found on Hugging Face. Check the spelling.", "red")
            return
        if error:
            self.show_detected("Couldn't reach Hugging Face. Please choose the type yourself.", "#b35c00")
            return
        guess, reason = guess_model_type(info)
        if guess is None:
            self.show_detected(f"Couldn't tell the type: {reason}. Please choose it yourself.", "#b35c00")
            return
        self.type_var.set(MODEL_TYPES[guess][0])
        self.update_examples()
        note = " ONNX is picked because it's faster; PyTorch gives the same answers." if guess == "zero_shot_onnx" else ""
        self.show_detected(f"Detected automatically: {reason}.{note} Change it if it's wrong.", "green")

    # ----- adding ----------------------------------------------------------------

    def submit(self) -> None:
        model_id = clean_model_id(self.id_var.get())
        if not re.fullmatch(r"[\w.-]+/[\w.-]+", model_id):
            messagebox.showwarning(
                "Add model", "Enter a Hugging Face model ID like  organization/model-name.", parent=self
            )
            return
        model_type = self.selected_type()
        key = "custom_" + re.sub(r"[^a-z0-9]+", "_", f"{model_id}_{model_type}".lower()).strip("_")
        if any(m["key"] == key for m in self.parent.models):
            messagebox.showinfo("Add model", "That model is already in the list with this type.", parent=self)
            return
        self.add_btn.configure(state="disabled")
        self.lookup(model_id, lambda mid: self.finish_submit(mid, model_type, key))

    def finish_submit(self, model_id: str, model_type: str, key: str) -> None:
        self.add_btn.configure(state="normal")
        info, error = self.hub_cache[model_id]
        if error == "not found":
            messagebox.showerror(
                "Add model",
                f"'{model_id}' was not found on Hugging Face.\n\nCheck the spelling. "
                "Private or gated models need an HF_TOKEN in the .env file.",
                parent=self,
            )
            return
        if error and not messagebox.askyesno(
            "Add model", f"Couldn't check the model: {error}.\n\nAdd it anyway?", parent=self
        ):
            return
        if info:
            guess, reason = guess_model_type(info)
            chosen = MODEL_TYPES[model_type][0]
            if guess and not same_family(guess, model_type):
                question = (f"This looks like a different type: {reason}.\n"
                            f"It will probably fail as '{chosen}'.\n\nAdd it anyway?")
            elif guess is None:
                question = (f"Couldn't confirm the type: {reason}.\n"
                            f"It may fail to load as '{chosen}'.\n\nAdd it anyway?")
            else:
                question = None
            if question and not messagebox.askyesno("Add model", question, parent=self):
                return

        self.parent.add_model({
            "key": key,
            "name": self.name_var.get().strip() or f"{model_id.split('/')[-1]} ({MODEL_TYPES[model_type][0].split(',')[0]})",
            "type": model_type,
            "model_id": model_id,
            "hint": self.hint_var.get().strip() or "added by you",
            "builtin": False,
        })
        self.destroy()


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class Dashboard(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1250x800")
        self.minsize(1000, 640)

        self.busy: str | None = None  # "run" or "setup" while a job is going
        self.proc: subprocess.Popen | None = None
        self.lines: queue.Queue = queue.Queue()
        self.stopped_by_user = False
        self.missing_module = False
        self.need_python = False
        self.started_at = 0.0
        self.done_questions = 0
        self.total_questions = 0
        self.results: dict | None = None
        self.models: list[dict] = load_models()
        self.dataset_entries: list[tuple[str, int, str | None]] = []  # (spec, count, error)

        style = ttk.Style(self)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("Start.TButton", font=("Segoe UI", 11, "bold"), padding=6)

        settings = self.load_settings()
        left = ttk.Frame(self, padding=10)
        left.pack(side="left", fill="y")
        right = ttk.Frame(self, padding=(0, 10, 10, 10))
        right.pack(side="left", fill="both", expand=True)

        self.build_models(left, settings)
        self.build_datasets(left, settings)
        self.build_options(left, settings)
        self.build_controls(left)
        self.build_output(right)

        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.after(100, self.poll)
        if RESULTS_FILE.is_file():
            self.load_results(RESULTS_FILE, switch_tab=False)
        self.after(300, self.check_environment_on_start)

    # ----- layout -----------------------------------------------------------

    def build_models(self, parent, settings) -> None:
        box = ttk.LabelFrame(parent, text="Models", padding=8)
        box.pack(fill="x")
        self.model_checks = ttk.Frame(box)
        self.model_checks.pack(fill="x")
        self.model_vars: dict[str, tk.BooleanVar] = {}
        self.render_models(settings.get("models", [m["key"] for m in self.models if m.get("builtin")]))

        row = ttk.Frame(box)
        row.pack(anchor="w", pady=(6, 0))
        ttk.Button(row, text="Select all", command=lambda: self.set_models(True)).pack(side="left")
        ttk.Button(row, text="Clear", command=lambda: self.set_models(False)).pack(side="left", padx=6)
        ttk.Button(row, text="Add model...", command=lambda: AddModelDialog(self)).pack(side="left")
        ttk.Button(row, text="Remove model...", command=self.remove_model_dialog).pack(side="left", padx=6)

    def render_models(self, selected: list[str]) -> None:
        for child in self.model_checks.winfo_children():
            child.destroy()
        self.model_vars = {}
        if not self.models:
            ttk.Label(self.model_checks, text=f"No models found in {MODELS_FILE.name}.", foreground="red").pack(
                anchor="w"
            )
        for m in self.models:
            var = tk.BooleanVar(value=m["key"] in selected)
            text = m["name"] + (f"  ({m['hint']})" if m.get("hint") else "")
            ttk.Checkbutton(self.model_checks, text=text, variable=var).pack(anchor="w", pady=1)
            self.model_vars[m["key"]] = var

    def build_datasets(self, parent, settings) -> None:
        box = ttk.LabelFrame(parent, text="Datasets (click to select one or more)", padding=8)
        box.pack(fill="both", expand=True, pady=10)
        frame = ttk.Frame(box)
        frame.pack(fill="both", expand=True)
        self.dataset_list = tk.Listbox(
            frame, selectmode="multiple", height=6, width=52, activestyle="none", exportselection=False
        )
        scroll = ttk.Scrollbar(frame, orient="vertical", command=self.dataset_list.yview)
        self.dataset_list.configure(yscrollcommand=scroll.set)
        self.dataset_list.pack(side="left", fill="both", expand=True)
        scroll.pack(side="left", fill="y")

        row = ttk.Frame(box)
        row.pack(anchor="w", pady=(6, 0))
        ttk.Button(row, text="Add dataset...", command=self.add_datasets).pack(side="left")
        ttk.Button(row, text="Refresh", command=self.refresh_datasets).pack(side="left", padx=6)
        ttk.Button(row, text="Open folder", command=lambda: self.open_path(DATASETS_DIR)).pack(side="left")

        self.refresh_datasets(settings.get("datasets", ["decision_choices"]))

    def build_options(self, parent, settings) -> None:
        box = ttk.LabelFrame(parent, text="Options", padding=8)
        box.pack(fill="x")
        self.runs_var = tk.IntVar(value=settings.get("runs", 3))
        self.threads_var = tk.IntVar(value=settings.get("threads", 0))
        self.offline_var = tk.BooleanVar(value=settings.get("offline", False))

        ttk.Label(box, text="Timed runs per question:").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(box, from_=1, to=20, width=6, textvariable=self.runs_var).grid(row=0, column=1, sticky="w", padx=6)
        ttk.Label(box, text="more = steadier timings, slower run", foreground="gray").grid(row=0, column=2, sticky="w")

        ttk.Label(box, text="CPU threads (PyTorch models):").grid(row=1, column=0, sticky="w", pady=4)
        ttk.Spinbox(box, from_=0, to=os.cpu_count() or 64, width=6, textvariable=self.threads_var).grid(
            row=1, column=1, sticky="w", padx=6
        )
        ttk.Label(box, text="0 = automatic", foreground="gray").grid(row=1, column=2, sticky="w")

        ttk.Checkbutton(
            box, text="Offline mode (use downloaded models only, no internet checks)", variable=self.offline_var
        ).grid(row=2, column=0, columnspan=3, sticky="w")

    def build_controls(self, parent) -> None:
        box = ttk.Frame(parent)
        box.pack(fill="x", pady=(10, 0))
        row = ttk.Frame(box)
        row.pack(fill="x")
        self.start_btn = ttk.Button(row, text="Start", style="Start.TButton", command=self.start)
        self.start_btn.pack(side="left", fill="x", expand=True)
        self.stop_btn = ttk.Button(row, text="Stop", command=self.stop, state="disabled")
        self.stop_btn.pack(side="left", padx=6)
        self.setup_btn = ttk.Button(row, text="Setup...", command=self.setup)
        self.setup_btn.pack(side="left")
        self.export_btn = ttk.Button(row, text="Export CSV...", command=self.export_csv, state="disabled")
        self.export_btn.pack(side="left", padx=(6, 0))

        self.progress = ttk.Progressbar(box, mode="determinate")
        self.progress.pack(fill="x", pady=(8, 2))
        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(box, textvariable=self.status_var, wraplength=440).pack(anchor="w")

    def build_output(self, parent) -> None:
        self.tabs = ttk.Notebook(parent)
        self.tabs.pack(fill="both", expand=True)

        # Results tab: summary table on top, per-question answers below.
        results = ttk.Frame(self.tabs, padding=8)
        self.tabs.add(results, text="Results")
        top = ttk.Frame(results)
        top.pack(fill="x")
        ttk.Label(top, text="Dataset:").pack(side="left")
        self.result_dataset = ttk.Combobox(top, state="readonly", width=40)
        self.result_dataset.pack(side="left", padx=6)
        self.result_dataset.bind("<<ComboboxSelected>>", lambda _e: self.show_results(self.result_dataset.get()))
        self.results_info = ttk.Label(top, text="No results yet. Choose models and datasets, then press Start.")
        self.results_info.pack(side="left", padx=10)

        ttk.Label(results, text="Summary (best accuracy first)", font=("Segoe UI", 10, "bold")).pack(
            anchor="w", pady=(10, 2)
        )
        self.summary_tree = ttk.Treeview(results, columns=SUMMARY_COLUMNS, show="headings", height=6)
        for col in SUMMARY_COLUMNS:
            self.summary_tree.heading(col, text=col)
            self.summary_tree.column(col, width=220 if col == "Model" else 105, anchor="w" if col == "Model" else "center")
        self.summary_tree.pack(fill="x")

        ttk.Label(results, text="Answers per question (✓ right, ✗ wrong)", font=("Segoe UI", 10, "bold")).pack(
            anchor="w", pady=(12, 2)
        )
        detail_frame = ttk.Frame(results)
        detail_frame.pack(fill="both", expand=True)
        self.detail_tree = ttk.Treeview(detail_frame, show="headings")
        ys = ttk.Scrollbar(detail_frame, orient="vertical", command=self.detail_tree.yview)
        xs = ttk.Scrollbar(detail_frame, orient="horizontal", command=self.detail_tree.xview)
        self.detail_tree.configure(yscrollcommand=ys.set, xscrollcommand=xs.set)
        self.detail_tree.grid(row=0, column=0, sticky="nsew")
        ys.grid(row=0, column=1, sticky="ns")
        xs.grid(row=1, column=0, sticky="ew")
        detail_frame.rowconfigure(0, weight=1)
        detail_frame.columnconfigure(0, weight=1)

        # Log tab: live output of the benchmark or setup.
        log_tab = ttk.Frame(self.tabs, padding=4)
        self.tabs.add(log_tab, text="Live log")
        self.log = ScrolledText(log_tab, font=("Consolas", 9), wrap="none", state="disabled")
        self.log.pack(fill="both", expand=True)

    # ----- models -------------------------------------------------------------

    def selected_models(self) -> list[str]:
        return [key for key, var in self.model_vars.items() if var.get()]

    def add_model(self, model: dict) -> None:
        selected = self.selected_models() + [model["key"]]
        self.models = load_models() + [model]
        save_models(self.models)
        self.render_models(selected)
        self.status_var.set(
            f"Added {model['name']}. It downloads the first time it runs, or press Setup... to download it now."
        )

    def remove_model_dialog(self) -> None:
        custom = [m for m in self.models if not m.get("builtin")]
        if not custom:
            messagebox.showinfo(
                "Remove model", "There are no added models to remove.\nThe 4 original models can only be unticked."
            )
            return
        win = tk.Toplevel(self)
        win.title("Remove model")
        win.transient(self)
        win.resizable(False, False)
        body = ttk.Frame(win, padding=14)
        body.pack(fill="both", expand=True)
        ttk.Label(body, text="Models you added:").pack(anchor="w")
        listbox = tk.Listbox(body, width=70, height=min(10, len(custom)), exportselection=False)
        for m in custom:
            listbox.insert("end", f"{m['name']}   [{m['model_id']}]")
        listbox.pack(fill="x", pady=6)
        listbox.selection_set(0)

        def remove() -> None:
            picked = listbox.curselection()
            if not picked:
                return
            model = custom[picked[0]]
            if not messagebox.askyesno(
                "Remove model",
                f"Remove {model['name']} from the list?\n\n(Its downloaded files stay in the Hugging Face cache.)",
                parent=win,
            ):
                return
            selected = [k for k in self.selected_models() if k != model["key"]]
            self.models = [m for m in load_models() if m["key"] != model["key"]]
            save_models(self.models)
            self.render_models(selected)
            self.status_var.set(f"Removed {model['name']}.")
            win.destroy()

        buttons = ttk.Frame(body)
        buttons.pack(anchor="e")
        ttk.Button(buttons, text="Remove", command=remove).pack(side="left")
        ttk.Button(buttons, text="Cancel", command=win.destroy).pack(side="left", padx=(6, 0))
        win.grab_set()

    # ----- datasets -----------------------------------------------------------

    def refresh_datasets(self, selected: list[str] | None = None) -> None:
        if selected is None:
            selected = self.selected_datasets(include_invalid=True)
        self.dataset_entries = [(BUILTIN_DATASET, 1, None)]
        if DATASETS_DIR.is_dir():
            for path in sorted(DATASETS_DIR.glob("*.json")):
                count, error = check_dataset(path)
                self.dataset_entries.append((path.stem, count, error))

        self.dataset_list.delete(0, "end")
        for i, (spec, count, error) in enumerate(self.dataset_entries):
            if spec == BUILTIN_DATASET:
                label = "default  (1 question, built-in example)"
            elif error:
                label = f"{spec}  (INVALID: {error})"
            else:
                label = f"{spec}  ({count} questions)"
            self.dataset_list.insert("end", label)
            if error:
                self.dataset_list.itemconfigure(i, foreground="red")
            if spec in selected:
                self.dataset_list.selection_set(i)

    def selected_datasets(self, include_invalid: bool = False) -> list[str]:
        picked = [self.dataset_entries[i] for i in self.dataset_list.curselection()]
        return [spec for spec, _count, error in picked if include_invalid or not error]

    def add_datasets(self) -> None:
        paths = filedialog.askopenfilenames(
            title="Add dataset JSON files", filetypes=[("JSON datasets", "*.json"), ("All files", "*.*")]
        )
        if not paths:
            return
        DATASETS_DIR.mkdir(exist_ok=True)
        added = []
        for raw in paths:
            src = Path(raw)
            _count, error = check_dataset(src)
            if error:
                messagebox.showerror(APP_TITLE, f"{src.name} is not a valid dataset:\n\n{error}")
                continue
            dest = DATASETS_DIR / src.name
            if dest.resolve() != src.resolve():
                if dest.exists() and not messagebox.askyesno(
                    APP_TITLE, f"{dest.name} already exists in datasets. Replace it?"
                ):
                    continue
                shutil.copy2(src, dest)
            added.append(dest.stem)
        self.refresh_datasets(self.selected_datasets(include_invalid=True) + added)
        if added:
            self.status_var.set(f"Added: {', '.join(added)}")

    # ----- environment check and setup -----------------------------------------

    def check_environment_on_start(self) -> None:
        if not SCRIPT.is_file():
            self.append_log(f"compare_decisions.py was not found next to the dashboard ({PROJECT_DIR}).\n")
            self.status_var.set("compare_decisions.py is missing. Keep the exe in the project folder.")
            self.tabs.select(1)
        elif not VENV_PYTHON.is_file():
            self.status_var.set("Python environment not set up yet. Press Setup...")
            if messagebox.askyesno(
                APP_TITLE,
                "The Python environment (.venv) for running the models isn't set up yet.\n\n"
                "Set it up now? This installs PyTorch and the other packages (about 1 GB, a few minutes).",
            ):
                self.start_job("setup", self.setup_worker, [])

    def setup(self) -> None:
        if self.busy:
            return
        selected = self.selected_models()
        answer = messagebox.askyesnocancel(
            "Setup",
            "Setup creates the Python environment if needed, installs any missing packages, "
            "and checks that they work.\n\n"
            f"Also download and test the {len(selected)} ticked model(s)?\n"
            "(Needed once before Offline mode can use them.)\n\n"
            "Yes = setup + download models\nNo = setup only\nCancel = do nothing",
        )
        if answer is None:
            return
        self.start_job("setup", self.setup_worker, selected if answer else [])

    def setup_worker(self, model_keys: list[str]) -> int:
        """Runs in a background thread. Each step's output goes to the Live log."""
        log = self.lines.put
        if not VENV_PYTHON.is_file():
            log("Looking for Python 3.10-3.13 (64-bit) on this PC...\n")
            base = find_base_python()
            if base is None:
                self.need_python = True
                log(
                    "\nNo suitable Python found.\n"
                    f"Install Python 3.13 (64-bit) from {PYTHON_DOWNLOAD_URL}\n"
                    "and tick 'Add python.exe to PATH' in the installer. Then press Setup... again.\n"
                )
                return 1
            log(f"Found {base}\n\nCreating the Python environment (.venv)...\n")
            if self.run_step([base, "-m", "venv", str(VENV_DIR)]) != 0:
                return 1

        py = str(VENV_PYTHON)
        if self.run_step([py, "-c", "import torch"], show=False) != 0:
            log("\nInstalling PyTorch (CPU build, about 120 MB)...\n")
            if self.run_step([py, "-m", "pip", "install", "torch", "--index-url", TORCH_CPU_INDEX]) != 0:
                return 1
        log("\nInstalling any missing packages from requirements.txt...\n")
        if self.run_step([py, "-m", "pip", "install", "--disable-pip-version-check", "-r", str(REQUIREMENTS)]) != 0:
            return 1
        log("\nChecking that every package imports...\n")
        if self.run_step([py, "-c", IMPORT_CHECK]) != 0:
            return 1
        if model_keys:
            log("\nDownloading (first time only) and test-loading the selected models...\n")
            return self.run_step([py, "-u", str(SCRIPT), "--prepare", "--models", *model_keys])
        return 0

    # ----- benchmark run ----------------------------------------------------------

    def start(self) -> None:
        if self.busy:
            return
        models = self.selected_models()
        datasets = self.selected_datasets()
        if not models:
            messagebox.showwarning(APP_TITLE, "Select at least one model.")
            return
        if not datasets:
            messagebox.showwarning(APP_TITLE, "Select at least one valid dataset.")
            return
        if not VENV_PYTHON.is_file():
            if messagebox.askyesno(APP_TITLE, "The Python environment isn't set up yet. Run Setup now?"):
                self.start_job("setup", self.setup_worker, [])
            return
        try:
            runs = max(1, int(self.runs_var.get()))
            threads = max(0, int(self.threads_var.get()))
        except (tk.TclError, ValueError):
            messagebox.showwarning(APP_TITLE, "Runs and CPU threads must be whole numbers.")
            return
        self.save_settings()

        dataset_args = [
            spec if spec == BUILTIN_DATASET else str(DATASETS_DIR / f"{spec}.json") for spec in datasets
        ]
        cmd = [
            str(VENV_PYTHON), "-u", str(SCRIPT),
            "--models", *models,
            "--dataset", *dataset_args,
            "--runs", str(runs),
            "--threads", str(threads),
            "--results-json", str(RESULTS_FILE),
        ]
        env = {"HF_HUB_OFFLINE": "1"} if self.offline_var.get() else {}

        RESULTS_FILE.unlink(missing_ok=True)
        counts = {spec: count for spec, count, _ in self.dataset_entries}
        self.total_questions = len(models) * sum(counts[s] for s in datasets)
        self.done_questions = 0
        self.progress.configure(maximum=max(1, self.total_questions), value=0)
        self.start_job("run", self.run_step, cmd, env)

    # ----- background jobs ------------------------------------------------------------

    def start_job(self, kind: str, worker, *args) -> None:
        self.busy = kind
        self.stopped_by_user = False
        self.missing_module = False
        self.need_python = False
        self.started_at = time.time()
        if kind == "setup":
            self.progress.configure(mode="indeterminate")
            self.progress.start(15)
        self.clear_log()
        for btn in (self.start_btn, self.setup_btn):
            btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.tabs.select(1)
        threading.Thread(target=self.job_thread, args=(kind, worker, args), daemon=True).start()

    def job_thread(self, kind: str, worker, args) -> None:
        try:
            code = worker(*args)
        except Exception as exc:  # report anything unexpected instead of hanging the UI
            self.lines.put(f"\nERROR: {type(exc).__name__}: {exc}\n")
            code = 1
        self.lines.put(("done", kind, code))

    def run_step(self, cmd: list[str], env_extra: dict | None = None, show: bool = True) -> int:
        """Run one command (from a background thread) and stream its output to the log."""
        if self.stopped_by_user:
            return 1
        if show:
            self.lines.put("\n> " + subprocess.list2cmdline(cmd) + "\n\n")
        env = os.environ.copy()
        env.update({"PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}, **(env_extra or {}))
        try:
            proc = subprocess.Popen(
                cmd, cwd=PROJECT_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", env=env, creationflags=NO_WINDOW,
            )
        except OSError as exc:
            self.lines.put(f"Could not start {cmd[0]}: {exc}\n")
            return 1
        self.proc = proc
        for line in proc.stdout:
            if show:
                self.lines.put(line)
        code = proc.wait()
        self.proc = None
        return code

    def stop(self) -> None:
        if self.busy:
            self.stopped_by_user = True
            if self.proc:
                self.proc.kill()
            self.status_var.set("Stopping...")

    def poll(self) -> None:
        try:
            while True:
                line = self.lines.get_nowait()
                if isinstance(line, tuple):
                    self.finished(line[1], line[2])
                    continue
                self.append_log(line)
                if "ModuleNotFoundError" in line:
                    self.missing_module = True
                if self.busy == "run" and PROGRESS_LINE.match(line):
                    self.done_questions += 1
                    self.progress.configure(value=self.done_questions)
        except queue.Empty:
            pass
        if self.busy == "run" and not self.stopped_by_user:
            self.status_var.set(
                f"Running... {elapsed_text(time.time() - self.started_at)}  |  "
                f"{self.done_questions}/{self.total_questions} questions answered"
            )
        elif self.busy == "setup" and not self.stopped_by_user:
            self.status_var.set(f"Setting up... {elapsed_text(time.time() - self.started_at)}  (see Live log)")
        self.after(100, self.poll)

    def finished(self, kind: str, code: int) -> None:
        elapsed = elapsed_text(time.time() - self.started_at)
        self.busy = None
        self.progress.stop()
        self.progress.configure(mode="determinate", value=0)
        for btn in (self.start_btn, self.setup_btn):
            btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")

        if self.stopped_by_user:
            self.status_var.set("Stopped.")
        elif kind == "setup":
            if code == 0:
                self.status_var.set(f"Setup finished in {elapsed}. Everything is ready.")
                self.append_log("\nSetup finished. Everything is ready.\n")
            else:
                self.status_var.set("Setup failed. See the Live log tab for details.")
                if self.need_python and messagebox.askyesno(
                    "Setup", "Python 3.10-3.13 (64-bit) is needed. Open the Python download page?"
                ):
                    webbrowser.open(PYTHON_DOWNLOAD_URL)
        elif code == 0 and RESULTS_FILE.is_file():
            self.progress.configure(maximum=1, value=1)
            self.status_var.set(f"Done in {elapsed}.")
            self.load_results(RESULTS_FILE)
        elif self.missing_module:
            self.status_var.set("Some Python packages are missing. Press Setup... to install them.")
        else:
            self.status_var.set(f"Run failed (exit code {code}). See the Live log tab.")

    # ----- results --------------------------------------------------------------

    def load_results(self, path: Path, switch_tab: bool = True) -> None:
        try:
            self.results = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        names = [ds["name"] for ds in self.results["datasets"]]
        self.result_dataset.configure(values=names)
        if names:
            self.result_dataset.set(names[0])
            self.show_results(names[0])
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(path.stat().st_mtime))
        self.results_info.configure(text=f"Run from {when}")
        self.export_btn.configure(state="normal")
        if switch_tab:
            self.tabs.select(0)

    def show_results(self, dataset_name: str) -> None:
        r = self.results
        self.summary_tree.delete(*self.summary_tree.get_children())
        rows = [s for s in r["summary"] if s["Dataset"] == dataset_name]
        rows.sort(key=lambda s: (-(s.get("Accuracy") if s.get("Accuracy") is not None else -1),
                                 s.get("Latency (ms)") or float("inf")))
        for s in rows:
            self.summary_tree.insert("", "end", values=[fmt(s.get(c)) for c in SUMMARY_COLUMNS])

        models = r["models"]
        columns = ["Question", "Expected"] + [m["key"] for m in models]
        self.detail_tree.delete(*self.detail_tree.get_children())
        self.detail_tree.configure(columns=columns, displaycolumns=columns)
        headings = {"Question": "Question", "Expected": "Expected", **{m["key"]: m["name"] for m in models}}
        for col in columns:
            self.detail_tree.heading(col, text=headings[col])
            self.detail_tree.column(col, width=200 if col == "Question" else 180, anchor="w", stretch=True)

        lookup = {(d["Item"], d["Model"]): d for d in r["details"] if d["Dataset"] == dataset_name}
        dataset = next(ds for ds in r["datasets"] if ds["name"] == dataset_name)
        for item in dataset["items"]:
            values = [item["id"], item["expected"] or "-"]
            for m in models:
                d = lookup.get((item["id"], m["key"]))
                if d is None:
                    values.append("-")
                else:
                    mark = {True: "✓ ", False: "✗ "}.get(d["Correct"], "")
                    values.append(mark + d["Choice"])
            self.detail_tree.insert("", "end", values=values)

    def export_csv(self) -> None:
        if not self.results:
            return
        target = filedialog.asksaveasfilename(
            title="Export results",
            initialdir=PROJECT_DIR / "results",
            initialfile="results.csv",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
        )
        if not target:
            return
        summary_path = Path(target)
        details_path = summary_path.with_name(summary_path.stem + "_details.csv")
        # utf-8-sig so Excel shows the text correctly.
        with summary_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=["Dataset"] + SUMMARY_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.results["summary"])
        with details_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=DETAIL_EXPORT_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.results["details"])
        self.status_var.set(f"Exported {summary_path.name} and {details_path.name}.")

    # ----- helpers --------------------------------------------------------------

    def set_models(self, value: bool) -> None:
        for var in self.model_vars.values():
            var.set(value)

    def append_log(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text)
        self.log.see("end")
        self.log.configure(state="disabled")

    def clear_log(self) -> None:
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    @staticmethod
    def open_path(path: Path) -> None:
        path.mkdir(exist_ok=True)
        os.startfile(path)

    def load_settings(self) -> dict:
        try:
            return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def save_settings(self) -> None:
        try:
            settings = {
                "models": self.selected_models(),
                "datasets": self.selected_datasets(include_invalid=True),
                "runs": self.runs_var.get(),
                "threads": self.threads_var.get(),
                "offline": self.offline_var.get(),
            }
            SETTINGS_FILE.write_text(json.dumps(settings, indent=2), encoding="utf-8")
        except (OSError, tk.TclError):
            pass

    def on_close(self) -> None:
        if self.busy:
            if not messagebox.askyesno(APP_TITLE, "A job is still running. Stop it and quit?"):
                return
            self.stop()
        self.save_settings()
        self.destroy()


if __name__ == "__main__":
    # Sharp text on high-DPI screens instead of Windows' blurry bitmap scaling.
    try:
        import ctypes

        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (AttributeError, OSError):
        pass
    Dashboard().mainloop()
