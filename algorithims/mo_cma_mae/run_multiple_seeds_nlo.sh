#!/bin/bash
set -euo pipefail
# Run MO-CMA-MAE (baseline, no surrogate) across multiple seeds for NLO
# Usage: bash run_multiple_seeds_nlo.sh [seed1 seed2 ...]
# Default seeds: 1 33 42 55 88

SEEDS=("${@:-1 33 42 55 88}")

echo "======================================================================"
echo "MO-CMA-MAE NLO — multi-seed run: ${SEEDS[*]}"
echo "======================================================================"

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "--- seed ${SEED} ---"
    bash "$(dirname "$0")/run_mo_cma_mae_nlo.sh" "${SEED}"
    EC=$?
    if [ $EC -ne 0 ]; then
        echo "ERROR: seed ${SEED} exited with code ${EC}" >&2
    fi
done

echo ""
echo "======================================================================"
echo "All seeds complete. Results:"
for SEED in "${SEEDS[@]}"; do
    echo "  algorithims/mo_cma_mae/mo_cma_mae_xtb_nlo_results_seed_${SEED}/"
done
echo "======================================================================"
