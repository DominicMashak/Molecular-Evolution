#!/bin/bash
set -euo pipefail
# Run NSGA-III to evolve crystals using SLICES encoding + QE DFT fitness.
# NSGA-III operates in the MatText-slices-2m UMAP embedding space.
# Pop size 15 = Das-Dennis ref_dirs for 2 objectives, n_partitions=4.
#
# Usage: bash run_nsga3_crystal.sh [SEED] [ELEMENT_SET]
#   SEED        - random seed (default: 42)
#   ELEMENT_SET - oxides | semiconductor | semiconductors | halide_perovskite
#                 (default: semiconductors)

SEED=${1:-42}
ELEMENT_SET=${2:-semiconductors}

PYTHON=/home/dominic/miniconda3/envs/mol-evo/bin/python
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/nsga3_crystal_${ELEMENT_SET}_seed_${SEED}"

export PYTHONPATH="$HOME/Molecular-Evolution/quantum_chemistry:$HOME/Molecular-Evolution/molev_utils:${PYTHONPATH:-}"

echo "NSGA-III crystal evolution"
echo "  Seed:        ${SEED}"
echo "  Element set: ${ELEMENT_SET}"
echo "  Encoder:     MatText-slices-2m (UMAP latent space)"
echo "  Output:      ${OUTPUT_DIR}"
echo ""

$PYTHON "$SCRIPT_DIR/main.py" \
    --encoding slices \
    --crystal-element-set "$ELEMENT_SET" \
    --fitness-mode qc \
    --objectives formation_energy bandgap \
    --optimize minimize maximize \
    --reference-point 5.0 0.0 \
    --embedding-model DeepChem/ChemBERTa-77M-MTR \
    --embedding-dims 8 \
    --embedding-device auto \
    --embedding-sample-size 500 \
    --n-partitions 4 \
    --sbx-eta 15 \
    --sbx-prob 0.9 \
    --pm-eta 20 \
    --pool-max-size 5000 \
    --n_gen 200 \
    --log_frequency 10 \
    --save_frequency 50 \
    --seed "$SEED" \
    --output_dir "$OUTPUT_DIR"
