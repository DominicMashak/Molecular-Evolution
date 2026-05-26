#!/bin/bash
set -euo pipefail
# Run Simulated Annealing to evolve semiconductor crystals using QE DFT fitness.
# Usage: bash run_crystal_qe.sh [SEED] [ELEMENT_SET]
#   SEED         - random seed (default: 42)
#   ELEMENT_SET  - oxides | semiconductor | semiconductors | halide_perovskite (default: semiconductors)

SEED=${1:-42}
ELEMENT_SET=${2:-semiconductors}

PYTHON=/home/dominic/miniconda3/envs/mol-evo/bin/python
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/sa_crystal_${ELEMENT_SET}_seed_${SEED}"

export PYTHONPATH="$HOME/Molecular-Evolution/quantum_chemistry:$PYTHONPATH"
export PYTHONPATH="$HOME/Molecular-Evolution/molev_utils:$PYTHONPATH"

echo "Simulated Annealing crystal evolution"
echo "  Seed:        ${SEED}"
echo "  Element set: ${ELEMENT_SET}"
echo "  Output:      ${OUTPUT_DIR}"
echo ""

$PYTHON "$SCRIPT_DIR/simulated_annealing.py" \
    --encoding slices \
    --crystal-element-set "$ELEMENT_SET" \
    --fitness-mode qc \
    --objective bandgap \
    --maximize \
    --n_iterations 2000 \
    --initial_temp 1.0 \
    --cooling_rate 0.995 \
    --log_frequency 100 \
    --seed "$SEED" \
    --output_dir "$OUTPUT_DIR"
