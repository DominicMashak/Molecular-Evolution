#!/bin/bash
set -euo pipefail
# MO-CMA-MAE: multi-objective CMA-MAE with CVT + ChemBERTa embeddings for NLO
# 4 objectives: beta/gamma ratio, energy/atom, alpha range distance, HOMO-LUMO gap distance
# Uses xTB (GFN2-xTB) oracle — no surrogate (baseline for SA-MO-CMA-MAE comparison)
# Compare against: run_sa_mo_cma_mae_nlo.sh
SEED=${1:-42}

echo "======================================================================"
echo "Running MO-CMA-MAE (CVT + embeddings) for NLO with seed ${SEED}..."
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
    --learning-rate 0.01 \
    --n-emitters 3 \
    --cma-batch-size 12 \
    --sigma0 0.5 \
    --objectives beta_gamma_ratio total_energy_atom_ratio alpha_range_distance homo_lumo_gap_range_distance \
    --optimize maximize minimize minimize minimize \
    --reference-point 0.0 0.0 500.0 100.0 \
    --pop_size 50 \
    --n_gen 200 \
    --log_frequency 10 \
    --save_frequency 50 \
    --seed ${SEED} \
    --output_dir mo_cma_mae_xtb_nlo_results_seed_${SEED}

echo ""
echo "======================================================================"
echo "MO-CMA-MAE NLO complete! Results in:"
echo "  mo_cma_mae_cvt_nlo_results_seed_${SEED}/"
echo "======================================================================"
