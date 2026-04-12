#!/bin/bash
set -euo pipefail

python ~/Molecular-Evolution/quantum_chemistry/main.py \
    --calculator cc \
    --functional "CCSD(T)" \
    --basis cc-pVDZ \
    --method full_tensor \
    --smiles "C1(-N)=C-C=C(-[N+](=O)-[O-])-C=C1" \
    --solvent none \
    --field-strength 0.001 \
    --properties beta