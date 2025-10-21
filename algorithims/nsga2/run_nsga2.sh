#!/bin/bash
cd ~/Molecular-Evolution/algorithims/nsga2

# Set Python path to include quantum_chemistry directory
export PYTHONPATH="$HOME/Molecular-Evolution/quantum_chemistry:$PYTHONPATH"

python3 main.py \
    --calculator dft \
    --functional HF \
    --basis STO-3G \
    --method full_tensor \
    --field-strength 0.001 \
    --pop_size 10 \
    --n_gen 100 \
    --output_dir nsga2_results \
    --objectives beta gamma natoms \
    --optimize maximize minimize minimize\
    --n-parents 5 \
    --n-children 5 \
    --no-stagnation-response \
    --seed 56 \