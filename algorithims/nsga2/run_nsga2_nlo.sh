#!/bin/bash
# Run NSGA-II for NLO (Non-Linear Optics) molecular optimization
# Uses quantum chemistry calculations to optimize hyperpolarizability

# Random seed (change this for different runs)
SEED=${1:-55}  # Default to 55 if no argument provided

echo "======================================================================"
echo "Running NSGA-II for NLO Optimization with seed ${SEED}..."
echo "======================================================================"

cd ~/Molecular-Evolution/algorithims/nsga2

# Set Python path to include quantum_chemistry directory
export PYTHONPATH="$HOME/Molecular-Evolution/quantum_chemistry:$PYTHONPATH"

python3 main.py \
    --fitness-mode qc \
    --calculator dft \
    --functional HF \
    --basis 3-21G \
    --method full_tensor \
    --field-strength 0.001 \
    --atom-set nlo \
    --pop_size 20 \
    --n_gen 3 \
    --output_dir nsga2_nlo_results_seed_${SEED} \
    --objectives beta_gamma_ratio total_energy_atom_ratio alpha_range_distance homo_lumo_gap_range_distance \
    --optimize maximize minimize minimize minimize \
    --reference-points 0.0 0.0 500.0 100.0 \
    --n-parents 10 \
    --n-children 20 \
    --no-stagnation-response \
    --seed ${SEED}

echo ""
echo "======================================================================"
echo "NLO optimization complete! Results saved to:"
echo "  nsga2_nlo_results_seed_${SEED}/"
echo "======================================================================"
