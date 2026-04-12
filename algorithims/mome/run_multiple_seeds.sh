#!/bin/bash
set -euo pipefail
# Run MOME with multiple random seeds
# Usage: ./run_multiple_seeds.sh [seed1] [seed2] [seed3] ...
# Example: ./run_multiple_seeds.sh 42 123 456 789 1000

# Default seeds if none provided
if [ $# -eq 0 ]; then
    SEEDS=(1 2 3 4 5 6 7 8 9)
    echo "No seeds provided. Using default seeds: ${SEEDS[@]}"
else
    SEEDS=("$@")
    echo "Running with seeds: ${SEEDS[@]}"
fi

# Run each seed
for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Running with seed ${SEED}"
    echo "=========================================="

    cd ~/Molecular-Evolution/algorithims/mome
    export PYTHONPATH="$HOME/Molecular-Evolution/quantum_chemistry:$PYTHONPATH"

    python main.py \
        --calculator dft \
        --functional HF \
        --basis 3-21G \
        --method full_tensor \
        --field-strength 0.001 \
        --pop_size 50 \
        --n_gen 100 \
        --objectives beta_gamma_ratio total_energy_atom_ratio alpha_range_distance homo_lumo_gap_range_distance \
        --optimize maximize minimize minimize minimize \
        --iterations_per_gen 20 \
        --log_frequency 1 \
        --save_frequency 1 \
        --output_dir mome_results_seed_${SEED} \
        --seed ${SEED} \
        --reference-point 0.0 0.0 500.0 100.0

    echo "Completed seed ${SEED}"
done

echo ""
echo "=========================================="
echo "All seeds completed!"
echo "=========================================="
echo ""
echo "Results directories:"
for SEED in "${SEEDS[@]}"; do
    echo "  mome_results_seed_${SEED}/"
done
