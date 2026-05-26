#!/bin/bash
set -euo pipefail
# Run CMA-ME on NLO molecules (quantum chemistry fitness).
# Usage: bash run_cma_me_nlo.sh [SEED]
#   SEED - random seed (default: 42)

SEED=${1:-42}

PYTHON=/home/dominic/miniconda3/envs/mol-evo/bin/python
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/cma_me_nlo_results_seed_${SEED}"

$PYTHON "$SCRIPT_DIR/main.py" \
    --seed "$SEED" \
    --atom-set nlo \
    --fitness-mode qc \
    --calculator dft \
    --functional HF \
    --basis 3-21G \
    --method full_tensor \
    --objective beta_mean \
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
