#!/bin/bash
set -euo pipefail
# Run (μ+λ) Evolution Strategy with multiple random seeds
# Usage: ./run_multiple_seeds.sh [seed1] [seed2] [seed3] ...
# Example: ./run_multiple_seeds.sh 42 123 456 789 1000

# Default seeds if none provided
if [ $# -eq 0 ]; then
    SEEDS=(31 32 33 34 35 36 37 38 39)
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

    python main.py \
        --calculator dft \
        --functional HF \
        --basis 3-21G \
        --method full_tensor \
        --field-strength 0.001 \
        --mu 20 \
        --lambda 20 \
        --n-gen 100 \
        --objective beta_gamma_ratio \
        --maximize \
        --output-dir mu_lambda_results_seed_${SEED} \
        --save-frequency 10 \
        --log-frequency 5 \
        --seed ${SEED} \
        --verbose

    echo "Completed seed ${SEED}"
done

echo ""
echo "=========================================="
echo "All seeds completed!"
echo "=========================================="
echo ""
echo "Results directories:"
for SEED in "${SEEDS[@]}"; do
    echo "  mu_lambda_results_seed_${SEED}/"
done

echo ""
echo "To generate plots for each run:"
for SEED in "${SEEDS[@]}"; do
    echo "  python performance.py mu_lambda_results_seed_${SEED}/"
done

echo ""
echo "To compare all runs:"
echo "  python -c \"from performance import plot_comparison; plot_comparison(['mu_lambda_results_seed_${SEEDS[0]}', 'mu_lambda_results_seed_${SEEDS[1]}', ...], labels=['Seed ${SEEDS[0]}', 'Seed ${SEEDS[1]}', ...], output_file='seed_comparison.png')\""
