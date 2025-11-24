#!/bin/bash
# Run NSGA-II with multiple random seeds
# Usage: ./run_multiple_seeds.sh [seed1] [seed2] [seed3] ...
# Example: ./run_multiple_seeds.sh 42 123 456 789 1000

# Default seeds if none provided
if [ $# -eq 0 ]; then
    SEEDS=(42 123 456 789 1000)
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

    cd ~/Molecular-Evolution/algorithims/nsga2
    export PYTHONPATH="$HOME/Molecular-Evolution/quantum_chemistry:$PYTHONPATH"

    python3 main.py \
        --calculator dft \
        --functional HF \
        --basis 3-21G \
        --method full_tensor \
        --field-strength 0.001 \
        --pop_size 20 \
        --n_gen 100 \
        --output_dir nsga2_results_seed_${SEED} \
        --objectives beta_gamma_ratio total_energy_atom_ratio alpha_range_distance homo_lumo_gap_range_distance \
        --optimize maximize minimize minimize minimize \
        --reference-points 0.0 0.0 500.0 100.0 \
        --n-parents 10 \
        --n-children 20 \
        --no-stagnation-response \
        --seed ${SEED}

    echo "Completed seed ${SEED}"
done

echo ""
echo "=========================================="
echo "All seeds completed!"
echo "=========================================="
echo ""
echo "Results directories:"
for SEED in "${SEEDS[@]}"; do
    echo "  nsga2_results_seed_${SEED}/"
done
