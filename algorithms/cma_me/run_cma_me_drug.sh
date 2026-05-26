#!/bin/bash
set -euo pipefail
# Run CMA-ME on drug-like molecules (SmartCADD fitness).
# Usage: bash run_cma_me_drug.sh [SEED]
#   SEED - random seed (default: 42)

SEED=${1:-42}

PYTHON=/home/dominic/miniconda3/envs/mol-evo/bin/python
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/cma_me_drug_results_seed_${SEED}"

$PYTHON "$SCRIPT_DIR/main.py" \
    --seed "$SEED" \
    --atom-set drug \
    --fitness-mode smartcadd \
    --smartcadd-mode descriptors \
    --objective qed \
    --maximize \
    --n-emitters 5 \
    --cma-batch-size 36 \
    --sigma0 0.5 \
    --n_gen 500 \
    --pop_size 100 \
    --log_frequency 10 \
    --save_frequency 50 \
    --measure-keys num_atoms num_bonds \
    --archive-dims 10 10 \
    --output_dir "$OUTPUT_DIR"
