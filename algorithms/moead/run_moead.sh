#!/bin/bash
# Run MOEA/D on NLO molecules (4-objective, DFT/HF/3-21G).
# ParallelMOEAD with auto PBI decomposition (n_obj=4 > 2).
# Pop size 84 = Das-Dennis ref_dirs for 4 objectives, n_partitions=6.
#
# Usage: bash run_moead.sh [SEED]
#   SEED defaults to 42

set -euo pipefail

SEED=${1:-42}
PYTHON=/home/dominic/miniconda3/envs/mol-evo/bin/python
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/moead_nlo_results_seed_${SEED}"

export PYTHONPATH="$HOME/Molecular-Evolution/quantum_chemistry:$HOME/Molecular-Evolution/molev_utils:${PYTHONPATH:-}"

$PYTHON "$SCRIPT_DIR/main.py" \
    --seed "$SEED" \
    --atom-set nlo \
    --fitness-mode qc \
    --calculator dft \
    --functional HF \
    --basis 3-21G \
    --method full_tensor \
    --field-strength 0.001 \
    --encoding smiles \
    --objectives beta_gamma_ratio total_energy_atom_ratio \
                 alpha_range_distance homo_lumo_gap_range_distance \
    --optimize maximize minimize minimize minimize \
    --reference-point 0.0 0.0 500.0 100.0 \
    --embedding-model DeepChem/ChemBERTa-77M-MTR \
    --embedding-dims 10 \
    --embedding-device auto \
    --embedding-sample-size 10000 \
    --n-partitions 6 \
    --n-neighbors 20 \
    --prob-neighbor-mating 0.9 \
    --pool-max-size 10000 \
    --n_gen 500 \
    --log_frequency 10 \
    --save_frequency 50 \
    --seed "$SEED" \
    --output_dir "$OUTPUT_DIR"
