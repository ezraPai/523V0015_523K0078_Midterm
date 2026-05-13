#!/usr/bin/env bash
set -euo pipefail

python -m src.train --config configs/mlp.yaml --experiment_name mlp_main
python -m src.train --config configs/lstm.yaml --experiment_name lstm_main
python -m src.train --config configs/bilstm_attn.yaml --experiment_name bilstm_attn_main
