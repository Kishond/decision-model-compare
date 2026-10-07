# Decision Model Compare

`compare_decisions.py` benchmarks local zero-shot decision / classification models side by side on datasets of choice questions. For each dataset it prints each model's answer to every question, then a summary table with accuracy, confidence and latency.

Everything runs locally. Weights download from the Hugging Face Hub on the first run and are cached after that. It needs no paid APIs and no background services such as Ollama.

## Models

The models are listed in `models.json`. Both the script and the dashboard read this file. The four built-in models are:

| Key | Model | Approach |
|-----|-------|----------|
| `local_onnx_deberta` | `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` | NLI zero-shot, exported to ONNX via Optimum |
| `local_pytorch_deberta` | same as above | NLI zero-shot, plain PyTorch pipeline (baseline) |
| `local_bge_reranker` | `BAAI/bge-reranker-large` | Cross-encoder scores each (state, option) pair, softmax over options |
| `local_gliclass` | `knowledgator/gliclass-small-v1.0` | GLiClass zero-shot classifier |

### Adding models

You can add any Hugging Face model that matches one of these types, with no code changes:

| `type` | Works with | Examples |
|--------|-----------|----------|
| `zero_shot_onnx` | NLI / zero-shot classification models, run with ONNX Runtime | `facebook/bart-large-mnli`, `MoritzLaurer/deberta-v3-large-zeroshot-v2.0` |
| `zero_shot_pytorch` | the same models, run with plain PyTorch | same as above |
| `reranker` | cross-encoder rerankers | `BAAI/bge-reranker-v2-m3`, `BAAI/bge-reranker-base` |
| `gliclass` | GLiClass models | `knowledgator/gliclass-base-v1.0` |

There are two ways to add one:

- **In the dashboard:** click **Add model...** and type or paste the model ID. The dashboard looks the model up on Hugging Face and fills in the type automatically; change it only if the guess is wrong. It warns you if the model doesn't exist or doesn't match the type you picked.
- **By hand:** add an entry to `models.json`:

  ```json
  {"key": "my_reranker", "name": "BGE Reranker Base", "type": "reranker", "model_id": "BAAI/bge-reranker-base", "hint": "smaller reranker"}
  ```

A model that works differently from these four types needs a new loader function. Add it to `MODEL_TYPES` in `compare_decisions.py`, and add the type's label to `MODEL_TYPES` in `dashboard.py`.

## Setup

The easiest way is the dashboard's **Setup...** button (see below). To set up by hand instead, use Python 3.10–3.13. Python 3.14 may not have wheels for torch or onnxruntime yet.

```powershell
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # optional: smaller CPU-only torch
pip install -r requirements.txt
```

You can also add a `.env` file, which is loaded through `python-dotenv`. Copy `.env.example` to get started:

```
HF_HOME=D:\hf-cache     # where model weights are cached
HF_TOKEN=hf_...         # optional; raises Hub download rate limits
```

## Dashboard

Double-click `DecisionDashboard.exe`, or run `.venv\Scripts\python.exe dashboard.py`.

- **Setup...**: checks the environment and repairs what's missing. It:
  - finds a 64-bit Python 3.10–3.13 on this PC (or offers to open the Python download page),
  - creates `.venv` if it doesn't exist,
  - installs CPU PyTorch and anything missing from `requirements.txt`,
  - checks that every package imports,
  - and can download and test-load the ticked models, so that Offline mode works later.

  If `.venv` is missing, the dashboard offers to run Setup when it opens.
- **Models**: tick the models to compare. **Add model...** and **Remove model...** manage the models you added. The 4 built-in models can only be unticked.
- **Datasets**: click datasets to select them (you can select several). **Add dataset...** checks a JSON file and copies it into `datasets/`. Invalid files are shown in red.
- **Options**: timed runs per question, CPU threads, and offline mode. Offline mode uses only downloaded models and skips internet checks.
- **Start / Stop**: the progress bar and **Live log** tab update while the benchmark runs. When it finishes, the **Results** tab shows a summary table and each model's answer per question (✓ / ✗).
- **Export CSV...**: saves a summary CSV and a per-question CSV.

The dashboard remembers your choices in `dashboard_settings.json`. It saves the last results in `results/last_run.json` and shows them again the next time it opens.

The exe contains only the dashboard window. It runs the models with the `.venv` Python in this folder, so keep the exe here. To rebuild the exe after changing `dashboard.py`, close the dashboard and run:

```powershell
powershell -ExecutionPolicy Bypass -File build_exe.ps1
```

## Run from the command line

Run these with the `.venv` Python, either after activating it or as `.venv\Scripts\python.exe compare_decisions.py`.

```powershell
python compare_decisions.py                                         # models in TARGET_MODELS, datasets in DATASETS
python compare_decisions.py --models local_onnx_deberta local_gliclass
python compare_decisions.py --dataset decision_choices my_set.json  # one or more datasets
python compare_decisions.py --dataset default                       # the single built-in test case
python compare_decisions.py --runs 5                                # timed runs per question
python compare_decisions.py --prepare --models local_gliclass       # only download + test-load models
```

To change the defaults, edit `TARGET_MODELS` and `DATASETS` at the top of the script.

The first run downloads about 3 GB of weights, and `bge-reranker-large` accounts for most of that. The first run also exports DeBERTa to ONNX, which can take a minute.

## Datasets

Datasets are JSON files. Put them in `datasets/` and refer to them by name (`--dataset decision_choices`), or pass any file path. The included `datasets/decision_choices.json` has 10 questions covering support routing, sentiment, urgency, intent and agent next-step decisions.

```json
{
  "name": "my_dataset",
  "hypothesis_template": "This message is a {}.",
  "items": [
    {
      "id": "double_billing",
      "state": "User message: 'I was double-billed, please refund me.'",
      "options": ["billing_issue", "technical_bug", "general_question"],
      "expected": "billing_issue",
      "hypothesis_template": "This ticket is about a {}."
    }
  ]
}
```

- `state` and `options` are required. Each question can have its own options.
- `expected` is optional. Questions without it are still run, but they don't count toward accuracy.
- `hypothesis_template` is optional at the dataset level and per question. It is the sentence each option is placed into, with underscores turned into spaces, when the DeBERTa and BGE models score it. Wording it to fit the question, such as `"The best next action is to {}."`, often changes the result a lot. GLiClass scores the option names directly and ignores the template.
- A bare JSON list of items also works.

The script checks every dataset before loading any model. It rejects files where `expected` isn't one of the options, or a template has no `{}`.

## Output

The per-question table shows each model's choice and marks wrong answers with `*`. The summary table has these columns:

- **Accuracy / Correct**: the share of questions with an `expected` answer that the model got right.
- **Avg Confidence**: the mean score of the chosen option, from 0 to 1. For the reranker, the score is a softmax over the options. GLiClass scores each option independently, so its values are often near 1 and only work as a ranking.
- **Latency (ms)**: the median time for one question, measured after one untimed warm-up call. **Min Latency** is the fastest call.
- **Load Time (s)**: the time to load the model, including any download or ONNX export.

If one model fails, it is listed with an error status and the other models still run.
