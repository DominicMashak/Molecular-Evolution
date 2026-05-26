#!/bin/bash
# Run MOEA/D with multiple random seeds (sequential).
#
# Usage: ./run_multiple_seeds.sh [seed1] [seed2] ...
#   Default: seeds 1 through 9

set -euo pipefail

if [ $# -eq 0 ]; then
    SEEDS=(1 2 3 4 5 6 7 8 9)
else
    SEEDS=("$@")
fi

PYTHON=/home/dominic/miniconda3/envs/mol-evo/bin/python
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PYTHONPATH="$HOME/Molecular-Evolution/quantum_chemistry:$HOME/Molecular-Evolution/molev_utils:${PYTHONPATH:-}"

for SEED in "${SEEDS[@]}"; do
    echo "=========================================="
    echo "Running MOEA/D with seed ${SEED}"
    echo "=========================================="
    "$PYTHON" "$SCRIPT_DIR/main.py" \
        --seed "$SEED" \
        --atom-set nlo \
        --fitness-mode qc \
        --calculator dft \
        --functional HF \
        --basis 3-21G \
        --method full_tensor \
        --field-strength 0.001 \
        --encoding smiles \
        --objectives beta_gamma_ratio total_energy_atom_ratio \
                     alpha_range_distance homo_lumo_gap_range_distance \
        --optimize maximize minimize minimize minimize \
        --reference-point 0.0 0.0 500.0 100.0 \
        --embedding-dims 10 \
        --embedding-device auto \
        --embedding-sample-size 10000 \
        --n-partitions 6 \
        --n-neighbors 20 \
        --prob-neighbor-mating 0.9 \
        --pool-max-size 10000 \
        --n_gen 500 \
        --log_frequency 10 \
        --save_frequency 50 \
        --output_dir "${SCRIPT_DIR}/moead_nlo_results_seed_${SEED}"
    echo "Seed ${SEED} complete."
done

echo "All seeds complete."
