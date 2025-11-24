#!/bin/bash
# Run MOME (Multi-Objective MAP-Elites) for molecular optimization

# Random seed (change this for different runs)
SEED=${1:-42}  # Default to 42 if no argument provided

echo "Running MOME with seed ${SEED}..."

cd ~/Molecular-Evolution/algorithims/mome

# Set Python path to include quantum_chemistry directory
export PYTHONPATH="$HOME/Molecular-Evolution/quantum_chemistry:$PYTHONPATH"

python main.py \
    --calculator dft \
    --functional HF \
    --basis 3-21G \
    --method full_tensor \
    --field-strength 0.001 \
    --pop_size 20 \
    --n_gen 100 \
    --objectives beta_gamma_ratio total_energy_atom_ratio alpha_range_distance homo_lumo_gap_range_distance \
    --optimize maximize minimize minimize minimize \
    --iterations_per_gen 20 \
    --log_frequency 1 \
    --save_frequency 1 \
    --output_dir mome_results_seed_${SEED} \
    --seed ${SEED} \
    --reference-point 0.0 0.0 500.0 100.0