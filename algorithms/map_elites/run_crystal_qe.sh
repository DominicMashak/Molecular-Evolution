#!/bin/bash
set -euo pipefail
# Run MAP-Elites to evolve semiconductor crystals using QE DFT fitness.
# Usage: bash run_crystal_qe.sh [SEED] [ELEMENT_SET]
#   SEED         - random seed (default: 42)
#   ELEMENT_SET  - oxides | semiconductor | semiconductors | halide_perovskite (default: semiconductors)

SEED=${1:-42}
ELEMENT_SET=${2:-semiconductors}

PYTHON=/home/dominic/miniconda3/envs/mol-evo/bin/python
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/map_elites_crystal_${ELEMENT_SET}_seed_${SEED}"

export PYTHONPATH="$HOME/Molecular-Evolution/quantum_chemistry:$PYTHONPATH"
export PYTHONPATH="$HOME/Molecular-Evolution/molev_utils:$PYTHONPATH"

echo "MAP-Elites crystal evolution"
echo "  Seed:        ${SEED}"
echo "  Element set: ${ELEMENT_SET}"
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
    --pop_size 30 \
    --n_gen 200 \
    --iterations_per_gen 10 \
    --log_frequency 10 \
    --save_frequency 50 \
    --seed "$SEED" \
    --output_dir "$OUTPUT_DIR"
