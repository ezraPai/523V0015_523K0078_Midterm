#!/usr/bin/env bash
set -euo pipefail

echo "Step 1/4: Preprocessing"
python -m src.preprocess --config configs/base.yaml

echo "Step 2/4: Main models"
bash scripts/run_baselines.sh

echo "Step 3/4: Controlled experiments"
bash scripts/run_ablations.sh

echo "Step 4/4: Ablation comparison plots"
python scripts/plot_ablation.py \
  --experiments exp_embed_dim_64 exp_embed_dim_128 exp_embed_dim_256 \
  --labels "emb_dim=64" "emb_dim=128" "emb_dim=256" \
  --title "Experiment 1: Embedding Dimension Ablation (Training Curves)" \
  --output report/figs/ablation_embed_dim.png
python scripts/plot_ablation.py \
  --experiments exp_dropout_0.2 exp_dropout_0.3 exp_dropout_0.5 \
  --labels "dropout=0.2" "dropout=0.3" "dropout=0.5" \
  --title "Experiment 2: Dropout Ablation (Training Curves)" \
  --output report/figs/ablation_dropout.png

echo "All done. Open notebooks/02_results_comparison.ipynb to build report tables."
