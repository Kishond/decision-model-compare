# CLAUDE.md

Notes for Claude Code sessions on this project. The user-facing docs are in README.md.

## What this is

A local benchmark that compares small "decision" models. A decision means picking one option out of a list for a given input. Everything runs on CPU with Hugging Face models: there are no paid APIs and no background services such as Ollama. Main target is Windows.

The user is not a developer. Explain things simply, step by step, with exact commands or clicks.

## Files

- `compare_decisions.py`: the benchmark (CLI).
  - Loads models from `models.json` and datasets from `datasets/*.json`.
  - Reports accuracy, confidence and latency per model, and saves the results to JSON with `--results-json`.
  - Other options: `--models`, `--dataset`, `--runs`, `--threads`, `--prepare` (download and test-load models only).
  - `TARGET_MODELS` and `DATASETS` at the top are the defaults.
- `models.json`: the model list shared by the script and the dashboard.
  - Each entry has `key`, `name`, `type`, `model_id`, `hint`, and `builtin`. The 4 originals have `builtin: true`.
  - Models the user adds through the dashboard are prefixed `custom_`.
- `dashboard.py`: Tkinter desktop app, standard library only.
  - It runs the benchmark as a subprocess with `.venv\Scripts\python.exe` and reads the results JSON back.
  - Features: Setup button, Add/Remove model (auto-detects the type from the Hugging Face API), dataset list, progress bar, results tables, CSV export.
  - It saves settings to `dashboard_settings.json` and the last run to `results/last_run.json`.
- `build_exe.ps1` / `Build-Dashboard.bat`: build `DecisionDashboard.exe` with PyInstaller.
  - On a fresh clone, they create `.venv` first.
  - The exe contains only the dashboard, about 11 MB. It must sit in the project folder.
- `datasets/`: `decision_choices.json` (10 questions) and `example_20.json` (20 questions).
  - Format: `{name, hypothesis_template, items: [{id, state, options, expected, hypothesis_template?}]}`.

## Model types

Each type is a different way of running a model. They are defined in `MODEL_TYPES`, which exists in both files; keep the two in sync:

| type | How it scores options |
|---|---|
| `zero_shot_onnx` | NLI zero-shot pipeline, exported to ONNX with optimum (`export=True`, re-exported on every load) |
| `zero_shot_pytorch` | same models, plain transformers pipeline |
| `reranker` | cross-encoder scores each (state, hypothesis) pair; softmax over options; uses the last logit column if there are 2+ |
| `gliclass` | gliclass pipeline, threshold 0, ignores hypothesis templates |

Options are humanized (`billing_issue` becomes "billing issue") and slotted into `hypothesis_template` for the NLI and reranker types. Wording the template to fit each question changes results a lot.

`guess_model_type()` in `dashboard.py` picks the type from the Hub's `pipeline_tag`, tags, and ID. It checks GLiClass first, then NLI, then reranker; NLI comes before reranker on purpose.

A model that fits none of the 4 types needs a new loader in `compare_decisions.py`. Example: `Cloudflare/clef`, a 27B multimodal model with custom code that is far too large for this PC.

## Environment

- Use Python 3.10–3.13, 64-bit. Python 3.14 is also installed on the main PC but is not used: torch and onnxruntime may lack wheels for it.
- `.venv` uses CPU-only torch (`--index-url https://download.pytorch.org/whl/cpu`), then `requirements.txt`.
- Run the script with `.venv\Scripts\python.exe`. Plain `python` points at the global 3.14, which has no packages.
- Model weights are cached in `~/.cache/huggingface/hub`. Set `HF_HOME` in `.env` to change that, and `HF_HUB_OFFLINE=1` to skip internet checks.
- Rebuilding the exe fails while `DecisionDashboard.exe` is running.

## Results so far (CPU, 14 threads)

- `decision_choices` (10 questions):
  - ONNX DeBERTa: 6/10, about 100 ms per question.
  - PyTorch DeBERTa: 6/10, about 1400 ms. It always gives the same answers as ONNX.
  - GLiClass: 5/10, about 38 ms. Its confidence saturates at 1.0, so treat it only as a ranking.
  - BGE reranker: 4/10, about 290 ms.
- `example_20`: the user ran it. ONNX DeBERTa scored 14/20; the other models' results weren't shared.
- All models miss "mixed" sentiment and the "ask a clarifying question" agent decisions.

## Testing notes

- The dashboard is tested by instantiating `Dashboard()`, calling `withdraw()`, and driving it from code.
- **Never take screen captures.** The user may be using the screen, for example playing a full-screen game.
- Don't pop windows up on the user's screen without asking.
- When a test touches `models.json`, back it up first and restore it afterwards. Delete the test's `dashboard_settings.json` and `results/last_run.json`.
- Not tested yet:
  - Setup on a machine with no Python, or with a brand-new `.venv` installing torch.
  - Anything on Mac.

## Open ideas the user may ask about

- Mac support. Needed: `.venv/bin/python` instead of `Scripts\python.exe`, a replacement for `os.startfile`, and `python3` instead of the `py` launcher.
- Cache the ONNX export so it isn't redone on every load.
- Show memory usage per model.
- Quick vs full run modes.
- Larger public datasets, for example ABCD customer support conversations.
- Compare with jev-arena (github.com/theaiautomators/jev-arena): a GPU and Docker based arena of decision models with per-kind runners in `workers/`.
- Suggested small models to try:
  - `MoritzLaurer/deberta-v3-base-zeroshot-v2.0`
  - `knowledgator/gliclass-base-v3.0` (may need a newer gliclass package)
  - `cross-encoder/ms-marco-MiniLM-L-6-v2`
  - `MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33`

## Git

- The repo is on GitHub. The user commits with TortoiseGit.
- `.gitignore` excludes `.venv/`, `.env`, `build/`, the exe, `results/` and `dashboard_settings.json`.
- Never commit Claude session files (`.jsonl`).
