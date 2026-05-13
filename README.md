# Text Sentiment Analysis — Final Project
## Course 503077 — Deep Learning, Semester 2 2025-2026

**Team:** 523V0015 · 523K0078

**Division of work:**
- 523V0015: Data preprocessing, MLP baseline, BiLSTM+Attention model, controlled experiments
- 523K0078: LSTM model, comparative analysis, error analysis, report writing

---

## Environment setup

```bash
pip install -r requirements.txt
```

Tested with Python 3.10+. A CUDA-capable GPU is optional;
the pipeline auto-detects GPU availability.

---

## How to reproduce results

### Step 1 — Preprocess the dataset
```bash
python src/preprocess.py
```
Downloads the IMDb dataset via HuggingFace and saves processed files
to `data/processed/`.

### Step 2 — Train all models and run ablations
```bash
bash scripts/run_all.sh
```
This trains MLP, LSTM, and BiLSTM+Attention, then runs all ablation
experiments. Checkpoints and logs are saved under `experiments/`.

Alternatively, train a single model:
```bash
bash scripts/run_baselines.sh
bash scripts/run_ablations.sh
```

### Step 3 — Generate results and analysis
Run the following notebooks in order using Jupyter:
1. `notebooks/01_eda.ipynb`          — EDA figures
2. `notebooks/02_results_comparison.ipynb` — comparison tables + learning curves
3. `notebooks/03_error_analysis.ipynb`     — error analysis

Outputs are saved to `report/figs/` and `report/tables/`.

---

## Report

`report/report.pdf` — final written report

---

## Project structure
configs/          — YAML config files (base + per-model + ablations)
data/             — Raw and processed dataset (auto-populated)
experiments/      — Training logs and checkpoints (auto-populated)
notebooks/        — EDA, results, and error analysis notebooks
report/           — Report PDF, figures, and tables
results/          — Serialized ablation results
scripts/          — Shell scripts to run the pipeline
src/              — Source code (preprocessing, models, training, evaluation)
requirements.txt  — Python dependencies
---

(End of README content)