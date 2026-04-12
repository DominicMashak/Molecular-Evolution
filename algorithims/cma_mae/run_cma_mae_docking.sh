#!/bin/bash
set -euo pipefail
# Run CMA-MAE optimising docking score against a specific protein target.
#
# SmartCADD pipeline:
#   1. ADMET/PAINS filter (fast, ~1ms)  — molecules failing are skipped
#   2. SMILES → 3D (RDKit/OpenBabel)
#   3. XTB geometry optimisation
#   4. Smina docking → docking_score (kcal/mol, lower = tighter binding)
#
# Usage: bash run_cma_mae_docking.sh [SEED] [PROTEIN_CODE]
#   SEED         - random seed (default: 42)
#   PROTEIN_CODE - PDB code of docking target (default: 6WPJ)
#
# Requirements: SmartCADD installed, xtb and smina on PATH.
# Set SMARTCADD_PATH if not auto-detected.

SEED=${1:-42}
PROTEIN_CODE=${2:-6WPJ}

PYTHON=/home/dominic/miniconda3/envs/mol-evo/bin/python
# Ensure conda env tools (smina, obabel, xtb) are on PATH for subprocesses
export PATH="/home/dominic/miniconda3/envs/mol-evo/bin:$PATH"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/cma_mae_docking_${PROTEIN_CODE}_seed_${SEED}"

$PYTHON "$SCRIPT_DIR/main.py" \
    --seed "$SEED" \
    --atom-set drug \
    --fitness-mode smartcadd \
    --smartcadd-mode docking \
    --protein-code "$PROTEIN_CODE" \
    --objective docking_score \
    --minimize \
    --problem drug_1obj_docking \
    --archive-type cvt \
    --cvt-measures embedding \
    --n-centroids 100 \
    --learning-rate 0.01 \
    --n-emitters 3 \
    --cma-batch-size 12 \
    --sigma0 0.5 \
    --n_gen 200 \
    --pop_size 50 \
    --log_frequency 10 \
    --save_frequency 50 \
    --output_dir "$OUTPUT_DIR"
