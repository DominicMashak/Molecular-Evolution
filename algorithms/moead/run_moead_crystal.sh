#!/bin/bash
# Run MOEA/D on crystals (3-objective, SLICES encoding, QE DFT fitness).
# Objectives: formation_energy (min), bandgap (max), effective_mass_e (min).
# ParallelMOEAD with auto PBI decomposition (n_obj=3).
# Uses MatText-slices-2m UMAP embedding space.
#
# Usage: bash run_moead_crystal.sh [SEED] [ELEMENT_SET]
#   SEED        - random seed (default: 42)
#   ELEMENT_SET - oxides | semiconductor | semiconductors | halide_perovskite
#                 (default: semiconductors)

set -euo pipefail

SEED=${1:-42}
ELEMENT_SET=${2:-semiconductors}

PYTHON=/home/dominic/miniconda3/envs/mol-evo/bin/python
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/moead_crystal_${ELEMENT_SET}_seed_${SEED}"

export PYTHONPATH="$HOME/Molecular-Evolution/quantum_chemistry:$HOME/Molecular-Evolution/molev_utils:${PYTHONPATH:-}"

echo "MOEA/D crystal evolution"
echo "  Seed:        ${SEED}"
echo "  Element set: ${ELEMENT_SET}"
echo "  Encoder:     MatText-slices-2m (UMAP latent space)"
echo "  Objectives:  formation_energy (min), bandgap (max), effective_mass_e (min)"
echo "  Output:      ${OUTPUT_DIR}"
echo ""

$PYTHON "$SCRIPT_DIR/main.py" \
    --encoding slices \
    --crystal-element-set "$ELEMENT_SET" \
    --fitness-mode qc \
    --objectives formation_energy bandgap effective_mass_e \
    --optimize minimize maximize minimize \
    --reference-point 5.0 0.0 10.0 \
    --embedding-model DeepChem/ChemBERTa-77M-MTR \
    --embedding-dims 8 \
    --embedding-device auto \
    --embedding-sample-size 500 \
    --n-partitions 4 \
    --n-neighbors 5 \
    --prob-neighbor-mating 0.9 \
    --pool-max-size 5000 \
    --n_gen 200 \
    --log_frequency 10 \
    --save_frequency 50 \
    --seed "$SEED" \
    --output_dir "$OUTPUT_DIR"
