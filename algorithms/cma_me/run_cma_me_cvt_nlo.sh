#!/bin/bash
set -euo pipefail
# CVT-CMA-ME with structural measures (num_atoms / num_bonds) for NLO optimization
SEED=${1:-42}

echo "======================================================================"
echo "Running CVT-CMA-ME (structural measures) for NLO with seed ${SEED}..."
echo "======================================================================"

PYTHON="$HOME/miniconda3/envs/mol-evo/bin/python"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$HOME/Molecular-Evolution/quantum_chemistry:$HOME/Molecular-Evolution/molev_utils:$PYTHONPATH"

$PYTHON "$SCRIPT_DIR/main.py" \
    --fitness-mode qc \
    --calculator dft \
    --functional HF \
    --basis 3-21G \
    --method full_tensor \
    --field-strength 0.001 \
    --atom-set nlo \
    --archive-type cvt \
    --n-centroids 100 \
    --cvt-measures structural \
    --cvt-samples 50000 \
    --n-emitters 5 \
    --cma-batch-size 36 \
    --sigma0 0.5 \
    --objective beta_gamma_ratio \
    --maximize \
    --pop_size 100 \
    --n_gen 500 \
    --log_frequency 10 \
    --save_frequency 50 \
    --seed ${SEED} \
    --output_dir "${SCRIPT_DIR}/cma_me_cvt_nlo_results_seed_${SEED}"

echo ""
echo "======================================================================"
echo "CVT-CMA-ME (structural) NLO complete! Results in:"
echo "  cma_me_cvt_nlo_results_seed_${SEED}/"
echo "======================================================================"
