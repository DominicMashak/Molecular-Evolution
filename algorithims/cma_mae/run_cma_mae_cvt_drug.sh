#!/bin/bash
set -euo pipefail
# CVT-CMA-MAE with structural measures for drug-like molecule optimization (SmartCADD)
SEED=${1:-42}

echo "======================================================================"
echo "Running CVT-CMA-MAE (structural measures) for drug with seed ${SEED}..."
echo "======================================================================"

cd ~/Molecular-Evolution/algorithims/cma_mae

export PYTHONPATH="$HOME/Molecular-Evolution/quantum_chemistry:$HOME/Molecular-Evolution/molev_utils:$PYTHONPATH"
PYTHON="$HOME/miniconda3/envs/mol-evo/bin/python"

$PYTHON main.py \
    --fitness-mode smartcadd \
    --smartcadd-mode descriptors \
    --atom-set drug \
    --encoding smiles \
    --archive-type cvt \
    --n-centroids 100 \
    --cvt-measures structural \
    --cvt-samples 50000 \
    --learning-rate 0.01 \
    --n-emitters 5 \
    --cma-batch-size 36 \
    --sigma0 0.5 \
    --objective qed \
    --maximize \
    --pop_size 100 \
    --n_gen 500 \
    --log_frequency 10 \
    --save_frequency 50 \
    --seed ${SEED} \
    --output_dir cma_mae_cvt_drug_results_seed_${SEED}

echo ""
echo "======================================================================"
echo "CVT-CMA-MAE (structural) drug complete! Results in:"
echo "  cma_mae_cvt_drug_results_seed_${SEED}/"
echo "======================================================================"
