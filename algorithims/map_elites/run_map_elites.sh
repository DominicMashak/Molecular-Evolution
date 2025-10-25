#!/bin/bash
cd ~/Molecular-Evolution/algorithims/map_elites

# Set Python path to include quantum_chemistry directory
export PYTHONPATH="$HOME/Molecular-Evolution/quantum_chemistry:$PYTHONPATH"

python main.py \
    --calculator dft \
    --functional HF \
    --basis STO-3G \
    --method full_tensor \
    --field-strength 0.001 \
    --pop_size 10 \
    --n_gen 10 \
    --iterations_per_gen 10 \
    --log_frequency 1 \
    --seed 42
