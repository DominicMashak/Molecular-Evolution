#!/bin/bash
set -euo pipefail
# Benchmark MAP-Elites and CMA-MAE on docking score optimisation.
#
# Runs both algorithms on 5 seeds sequentially (one at a time to avoid
# CPU contention on i7-12700K — see project memory).
#
# Usage: bash benchmark_docking.sh [PROTEIN_CODE]
#   PROTEIN_CODE - PDB code of docking target (default: 1AQ1)
#
# Results land in:
#   algorithims/map_elites/map_elites_docking_<CODE>_seed_<N>/
#   algorithims/cma_mae/cma_mae_docking_<CODE>_seed_<N>/

PROTEIN_CODE=${1:-6WPJ}
SEEDS=(1 2 3 4 5)

# Ensure conda env tools (smina, obabel, xtb) are on PATH for all subprocesses
export PATH="/home/dominic/miniconda3/envs/mol-evo/bin:$PATH"

MAP_ELITES_SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")/map_elites" && pwd)/run_map_elites_docking.sh"
CMA_MAE_SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")/cma_mae" && pwd)/run_cma_mae_docking.sh"

echo "========================================================"
echo "Docking benchmark: protein=${PROTEIN_CODE}, seeds=${SEEDS[*]}"
echo "========================================================"

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "-------- MAP-Elites | seed=${SEED} | protein=${PROTEIN_CODE} --------"
    bash "$MAP_ELITES_SCRIPT" "$SEED" "$PROTEIN_CODE"

    echo ""
    echo "-------- CMA-MAE    | seed=${SEED} | protein=${PROTEIN_CODE} --------"
    bash "$CMA_MAE_SCRIPT" "$SEED" "$PROTEIN_CODE"
done

echo ""
echo "========================================================"
echo "Benchmark complete."
echo "========================================================"
