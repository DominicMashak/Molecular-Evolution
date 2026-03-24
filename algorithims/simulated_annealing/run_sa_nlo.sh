#!/bin/bash
# Run Simulated Annealing for NLO (Non-Linear Optics) molecular optimization
# Uses quantum chemistry calculations

# Random seed (change this for different runs)
SEED=${1:-42}  # Default to 42 if no argument provided

echo "======================================================================"
echo "Running Simulated Annealing for NLO Optimization with seed ${SEED}..."
echo "======================================================================"

cd ~/Molecular-Evolution/algorithims/simulated_annealing

# Set Python path
export PYTHONPATH="$HOME/Molecular-Evolution:$PYTHONPATH"

python simulated_annealing.py \
    --fitness-mode qc \
    --calculator dft \
    --functional HF \
    --basis 3-21G \
    --method full_tensor \
    --field-strength 0.001 \
    --atom-set nlo \
    --objective beta_gamma_ratio \
    --maximize \
    --T_initial 100.0 \
    --T_min 0.01 \
    --cooling_rate 0.95 \
    --max_iterations 285 \
    --cooling_schedule linear \
    --output_dir sa_nlo_results_seed_${SEED} \
    --seed ${SEED}

echo ""
echo "======================================================================"
echo "NLO SA optimization complete! Results saved to:"
echo "  sa_nlo_results_seed_${SEED}/"
echo "======================================================================"
