#!/bin/bash
set -euo pipefail
# Run CMA-ME with multiple random seeds (NLO / drug descriptor mode).
# Usage: ./run_multiple_seeds.sh [PROBLEM] [seed1] [seed2] ...
#   PROBLEM - 'nlo' or 'drug' (default: nlo)
#   seeds   - list of integer seeds (default: 1 2 3 4 5)
#
# Examples:
#   ./run_multiple_seeds.sh nlo 42 123 456
#   ./run_multiple_seeds.sh drug 1 2 3 4 5

PROBLEM=${1:-nlo}
shift || true  # remaining args are seeds (empty is fine)

if [ $# -eq 0 ]; then
    SEEDS=(1 2 3 4 5)
    echo "No seeds provided. Using default seeds: ${SEEDS[*]}"
else
    SEEDS=("$@")
    echo "Running with seeds: ${SEEDS[*]}"
fi

PYTHON=/home/dominic/miniconda3/envs/mol-evo/bin/python
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$HOME/Molecular-Evolution/quantum_chemistry:$HOME/Molecular-Evolution/molev_utils:$PYTHONPATH"

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Problem: ${PROBLEM}  |  Seed: ${SEED}"
    echo "=========================================="

    if [ "$PROBLEM" = "drug" ]; then
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
    else
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
    fi

    echo "Completed ${PROBLEM} seed ${SEED}"
done

echo ""
echo "=========================================="
echo "All seeds completed!"
echo "=========================================="
echo ""
echo "Results directories:"
for SEED in "${SEEDS[@]}"; do
    echo "  cma_me_${PROBLEM}_results_seed_${SEED}/"
done
