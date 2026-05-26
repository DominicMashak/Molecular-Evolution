#!/bin/bash
set -euo pipefail
# Run NSGA-II to evolve semiconductor crystals using QE DFT fitness.
# Multi-objective: maximise bandgap, minimise formation energy.
# Usage: bash run_crystal_qe.sh [SEED] [ELEMENT_SET]
#   SEED         - random seed (default: 42)
#   ELEMENT_SET  - oxides | semiconductor | semiconductors | halide_perovskite (default: semiconductors)

SEED=${1:-42}
ELEMENT_SET=${2:-semiconductors}

PYTHON=/home/dominic/miniconda3/envs/mol-evo/bin/python
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/nsga2_crystal_${ELEMENT_SET}_seed_${SEED}"

export PYTHONPATH="$HOME/Molecular-Evolution/quantum_chemistry:$PYTHONPATH"
export PYTHONPATH="$HOME/Molecular-Evolution/molev_utils:$PYTHONPATH"

echo "NSGA-II crystal evolution"
echo "  Seed:        ${SEED}"
echo "  Element set: ${ELEMENT_SET}"
echo "  Output:      ${OUTPUT_DIR}"
echo ""

$PYTHON "$SCRIPT_DIR/main.py" \
    --encoding slices \
    --crystal-element-set "$ELEMENT_SET" \
    --fitness-mode qc \
    --objectives bandgap formation_energy \
    --optimize maximize minimize \
    --reference-points 0.0 0.0 \
    --n-parents 20 \
    --n-children 40 \
    --n_gen 200 \
    --no-stagnation-response \
    --seed "$SEED" \
    --output_dir "$OUTPUT_DIR"
