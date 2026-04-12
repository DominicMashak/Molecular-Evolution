#!/bin/bash
set -euo pipefail
# Run NSGA-II for molecular optimization

# Random seed (change this for different runs)
SEED=${1:-55}  # Default to 42 if no argument provided

echo "Running NSGA-II with seed ${SEED}..."

cd ~/Molecular-Evolution/algorithims/nsga2

# Set Python path to include quantum_chemistry directory
export PYTHONPATH="$HOME/Molecular-Evolution/quantum_chemistry:$PYTHONPATH"

python3 main.py \
    --calculator dft \
    --functional HF \
    --basis 3-21G \
    --method full_tensor \
    --field-strength 0.001 \
    --pop_size 20 \
    --n_gen 3 \
    --output_dir nsga2_results_seed_${SEED} \
    --objectives beta_gamma_ratio total_energy_atom_ratio alpha_range_distance homo_lumo_gap_range_distance \
    --optimize maximize minimize minimize minimize \
    --reference-points 0.0 0.0 500.0 100.0 \
    --n-parents 10 \
    --n-children 20 \
    --no-stagnation-response \
    --seed ${SEED}