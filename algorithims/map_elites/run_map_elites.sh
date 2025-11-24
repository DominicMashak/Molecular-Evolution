#!/bin/bash
# Run MAP-Elites for molecular optimization

# Random seed (change this for different runs)
SEED=${1:-42}  # Default to 42 if no argument provided

echo "Running MAP-Elites with seed ${SEED}..."

cd ~/Molecular-Evolution/algorithims/map_elites

# Set Python path to include quantum_chemistry directory
export PYTHONPATH="$HOME/Molecular-Evolution/quantum_chemistry:$PYTHONPATH"

python main.py \
    --calculator dft \
    --functional HF \
    --basis 3-21G \
    --method full_tensor \
    --field-strength 0.001 \
    --pop_size 30 \
    --n_gen 20 \
    --iterations_per_gen 30 \
    --log_frequency 1 \
    --save_frequency 1 \
    --output_dir map_elites_results_seed_${SEED} \
    --seed ${SEED}