#!/bin/bash
# Run Simulated Annealing with multiple random seeds
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

    cd ~/Molecular-Evolution/algorithims/simulated_annealing

    # Set Python path to include quantum_chemistry directory
    export PYTHONPATH="$HOME/Molecular-Evolution/quantum_chemistry:$PYTHONPATH"

    python simulated_annealing.py \
        --calculator dft \
        --functional HF \
        --basis 3-21G \
        --method full_tensor \
        --field-strength 0.001 \
        --initial_smiles "C1=CC=CC=C1" \
        --T_initial 100.0 \
        --T_min 0.01 \
        --cooling_rate 0.95 \
        --max_iterations 285 \
        --seed ${SEED} \
        --cooling_schedule linear \
        --output_dir sa_results_seed_${SEED}

    echo "Completed seed ${SEED}"
done

echo ""
echo "=========================================="
echo "All seeds completed!"
echo "=========================================="
echo ""
echo "Results directories:"
for SEED in "${SEEDS[@]}"; do
    echo "  sa_results_seed_${SEED}/"
done
