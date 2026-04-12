#!/bin/bash
<<<<<<< Updated upstream
=======
set -euo pipefail
>>>>>>> Stashed changes
# Run MAP-Elites for NLO (Non-Linear Optics) molecular optimization
# Uses quantum chemistry calculations with quality-diversity archive

# Random seed (change this for different runs)
SEED=${1:-42}  # Default to 42 if no argument provided

echo "======================================================================"
echo "Running MAP-Elites for NLO Optimization with seed ${SEED}..."
echo "======================================================================"

cd ~/Molecular-Evolution/algorithims/map_elites

# Set Python path to include quantum_chemistry directory
export PYTHONPATH="$HOME/Molecular-Evolution/quantum_chemistry:$PYTHONPATH"

python main.py \
    --fitness-mode qc \
    --calculator dft \
    --functional HF \
    --basis 3-21G \
    --method full_tensor \
    --field-strength 0.001 \
    --atom-set nlo \
    --objective-key beta_gamma_ratio \
    --pop_size 30 \
    --n_gen 20 \
    --iterations_per_gen 30 \
    --log_frequency 1 \
    --save_frequency 1 \
    --output_dir map_elites_nlo_results_seed_${SEED} \
    --seed ${SEED}

echo ""
echo "======================================================================"
echo "NLO MAP-Elites optimization complete! Results saved to:"
echo "  map_elites_nlo_results_seed_${SEED}/"
echo "======================================================================"
