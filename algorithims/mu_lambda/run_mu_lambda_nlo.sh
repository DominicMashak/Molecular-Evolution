#!/bin/bash
<<<<<<< Updated upstream
=======
set -euo pipefail
>>>>>>> Stashed changes
# Run (μ+λ) Evolution Strategy for NLO (Non-Linear Optics) molecular optimization
# Uses quantum chemistry calculations with single-objective optimization

# Random seed (change this for different runs)
SEED=${1:-42}  # Default to 42 if no argument provided

echo "======================================================================"
echo "Running (μ+λ) ES for NLO Optimization with seed ${SEED}..."
echo "======================================================================"

cd ~/Molecular-Evolution/algorithims/mu_lambda

# Set Python path
export PYTHONPATH="$HOME/Molecular-Evolution:$PYTHONPATH"

python main.py \
    --fitness-mode qc \
    --calculator dft \
    --functional HF \
    --basis 3-21G \
    --method full_tensor \
    --field-strength 0.001 \
    --atom-set nlo \
    --objective beta_gamma_ratio \
    --maximize \
    --mu 20 \
    --lambda 20 \
    --n-gen 100 \
    --save-frequency 10 \
    --log-frequency 5 \
    --output-dir mu_lambda_nlo_results_seed_${SEED} \
    --seed ${SEED} \
    --verbose

echo ""
echo "======================================================================"
echo "NLO (μ+λ) optimization complete! Results saved to:"
echo "  mu_lambda_nlo_results_seed_${SEED}/"
echo "======================================================================"
