#!/bin/bash
set -euo pipefail
# Run Simulated Annealing for molecular optimization

# Random seed (change this for different runs)
SEED=${1:-42}  # Default to 42 if no argument provided

echo "Running Simulated Annealing with seed ${SEED}..."

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
    --max_iterations 100 \
    --seed ${SEED} \
    --cooling_schedule linear \
    --output_dir sa_results_seed_${SEED}