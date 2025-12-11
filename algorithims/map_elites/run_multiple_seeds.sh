#!/bin/bash
# Run MAP-Elites with multiple random seeds
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

    cd ~/Molecular-Evolution/algorithims/map_elites
    export PYTHONPATH="$HOME/Molecular-Evolution/quantum_chemistry:$PYTHONPATH"

    python main.py \
        --calculator dft \
        --functional HF \
        --basis 3-21G \
        --method full_tensor \
        --field-strength 0.001 \
        --pop_size 50 \
        --n_gen 100 \
        --iterations_per_gen 20 \
        --log_frequency 1 \
        --save_frequency 1 \
        --output_dir map_elites_results_seed_${SEED} \
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
    echo "  map_elites_results_seed_${SEED}/"
done
