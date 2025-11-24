#!/bin/bash
# Run (μ+λ) Evolution for molecular optimization

# Random seed (change this for different runs)
SEED=${1:-42}  # Default to 42 if no argument provided

echo "Running (μ+λ) Evolution Strategy with seed ${SEED}..."

python main.py \
    --calculator dft \
    --functional HF \
    --basis 3-21G \
    --method full_tensor \
    --field-strength 0.001 \
    --mu 4 \
    --lambda 8 \
    --n-gen 10 \
    --objective beta_gamma_ratio \
    --maximize \
    --output-dir mu_lambda_results_seed_${SEED} \
    --save-frequency 1 \
    --log-frequency 1 \
    --seed ${SEED}