#!/bin/bash
set -euo pipefail
# SA-MO-CMA-MAE: Surrogate-Assisted Multi-Objective CMA-MAE for NLO
# Uses xTB (GFN2-xTB) oracle — fast (~5-30s/mol vs HF ~5-15min/mol)
# Surrogate pre-screens CMA-ES proposals (top 20%) before xTB evaluation
#
# Usage:
#   bash run_sa_mo_cma_mae_nlo.sh [SEED]
#
# To compare without surrogate (baseline):
#   bash run_mo_cma_mae_nlo.sh [SEED]

SEED=${1:-42}

echo "======================================================================"
echo "Running SA-MO-CMA-MAE (xTB oracle, surrogate) for NLO — seed ${SEED}"
echo "======================================================================"

cd ~/Molecular-Evolution/algorithims/mo_cma_mae

export PYTHONPATH="$HOME/Molecular-Evolution/quantum_chemistry:$HOME/Molecular-Evolution/molev_utils:$PYTHONPATH"
PYTHON="$HOME/miniconda3/envs/mol-evo/bin/python"

$PYTHON main.py \
    --fitness-mode qc \
    --calculator xtb \
    --xtb-method GFN2-xTB \
    --method full_tensor \
    --field-strength 0.001 \
    --atom-set nlo \
    --encoding selfies \
    --n-centroids 100 \
    --cvt-measures embedding \
    --cvt-samples 50000 \
    --embedding-dims 10 \
    --embedding-sample-size 10000 \
    --embedding-device auto \
    --objectives beta_gamma_ratio total_energy_atom_ratio alpha_range_distance homo_lumo_gap_range_distance \
    --optimize maximize minimize minimize minimize \
    --reference-point 0.0 0.0 500.0 100.0 \
    --learning-rate 0.01 \
    --n-emitters 3 \
    --cma-batch-size 12 \
    --sigma0 0.5 \
    --surrogate-type mlp \
    --surrogate-screen-fraction 0.2 \
    --surrogate-warmup 30 \
    --surrogate-retrain-interval 10 \
    --surrogate-beta 2.0 \
    --pop_size 50 \
    --n_gen 200 \
    --log_frequency 10 \
    --save_frequency 50 \
    --seed ${SEED} \
    --output_dir sa_mo_cma_mae_xtb_nlo_results_seed_${SEED}

echo ""
echo "======================================================================"
echo "SA-MO-CMA-MAE NLO complete! Results in:"
echo "  sa_mo_cma_mae_xtb_nlo_results_seed_${SEED}/"
echo "======================================================================"
