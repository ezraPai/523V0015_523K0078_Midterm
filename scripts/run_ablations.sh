#!/usr/bin/env bash
set -euo pipefail

echo "========================================"
echo "Embedding dimension ablation"
echo "========================================"
python -m src.train --config configs/exp_embed_dim_64.yaml --experiment_name exp_embed_dim_64
python -m src.train --config configs/exp_embed_dim_128.yaml --experiment_name exp_embed_dim_128
python -m src.train --config configs/exp_embed_dim_256.yaml --experiment_name exp_embed_dim_256

echo "========================================"
echo "Dropout ablation"
echo "========================================"
python -m src.train --config configs/exp_dropout_0.2.yaml --experiment_name exp_dropout_0.2
python -m src.train --config configs/exp_dropout_0.3.yaml --experiment_name exp_dropout_0.3
python -m src.train --config configs/exp_dropout_0.5.yaml --experiment_name exp_dropout_0.5
