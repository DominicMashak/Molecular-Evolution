#!/bin/bash
set -euo pipefail
# Resume docking benchmark from a given seed.
# Usage: bash benchmark_docking_resume.sh [PROTEIN_CODE] [START_SEED] [END_SEED]

PROTEIN_CODE=${1:-6WPJ}
START_SEED=${2:-2}
END_SEED=${3:-30}

export PATH="/home/dominic/miniconda3/envs/mol-evo/bin:$PATH"

MAP_ELITES_SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")/map_elites" && pwd)/run_map_elites_docking.sh"
CMA_ME_SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")/cma_me" && pwd)/run_cma_me_docking.sh"
CMA_MAE_SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")/cma_mae" && pwd)/run_cma_mae_docking.sh"

echo "========================================================"
echo "Docking benchmark (resume): protein=${PROTEIN_CODE}, seeds=${START_SEED}-${END_SEED}"
echo "Algorithms: MAP-Elites, CMA-ME, CMA-MAE  |  Encoding: SELFIES"
echo "========================================================"

for SEED in $(seq "$START_SEED" "$END_SEED"); do
    echo ""
    echo "-------- MAP-Elites | seed=${SEED} | protein=${PROTEIN_CODE} --------"
    bash "$MAP_ELITES_SCRIPT" "$SEED" "$PROTEIN_CODE"

    echo ""
    echo "-------- CMA-ME     | seed=${SEED} | protein=${PROTEIN_CODE} --------"
    bash "$CMA_ME_SCRIPT" "$SEED" "$PROTEIN_CODE"

    echo ""
    echo "-------- CMA-MAE    | seed=${SEED} | protein=${PROTEIN_CODE} --------"
    bash "$CMA_MAE_SCRIPT" "$SEED" "$PROTEIN_CODE"
done

echo ""
echo "========================================================"
echo "Benchmark complete."
echo "========================================================"
