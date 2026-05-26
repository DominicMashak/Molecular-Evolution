#!/bin/bash
set -euo pipefail
# Run CMA-ME to evolve semiconductor crystals using QE DFT fitness.
# CMA-ES operates in the MatText-slices-2m UMAP embedding space.
# Usage: bash run_crystal_qe.sh [SEED] [ELEMENT_SET]
#   SEED         - random seed (default: 42)
#   ELEMENT_SET  - oxides | semiconductor | semiconductors | halide_perovskite (default: semiconductors)

SEED=${1:-42}
ELEMENT_SET=${2:-semiconductors}

PYTHON=/home/dominic/miniconda3/envs/mol-evo/bin/python
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/cma_me_crystal_${ELEMENT_SET}_seed_${SEED}"

export PYTHONPATH="$HOME/Molecular-Evolution/quantum_chemistry:$PYTHONPATH"
export PYTHONPATH="$HOME/Molecular-Evolution/molev_utils:$PYTHONPATH"

echo "CMA-ME crystal evolution"
echo "  Seed:        ${SEED}"
echo "  Element set: ${ELEMENT_SET}"
echo "  Encoder:     MatText-slices-2m (UMAP latent space)"
echo "  Output:      ${OUTPUT_DIR}"
echo ""

$PYTHON "$SCRIPT_DIR/main.py" \
    --encoding slices \
    --crystal-element-set "$ELEMENT_SET" \
    --fitness-mode qc \
    --objective bandgap \
    --maximize \
    --measure-keys n_sites n_species \
    --measure-bounds 2 20 1 6 \
    --archive-dims 10 10 \
    --embedding-dims 8 \
    --embedding-sample-size 500 \
    --n-emitters 5 \
    --cma-batch-size 36 \
    --sigma0 0.5 \
    --pop_size 50 \
    --n_gen 200 \
    --log_frequency 10 \
    --save_frequency 50 \
    --seed "$SEED" \
    --output_dir "$OUTPUT_DIR"
