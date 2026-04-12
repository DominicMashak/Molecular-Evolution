#!/bin/bash
<<<<<<< Updated upstream
# Run CMA-MAE on drug-like molecules (SmartCADD fitness).
# Usage: bash run_cma_mae_drug.sh [SEED] [VAE_PATH]
#   SEED     - random seed (default: 42)
#   VAE_PATH - path to pre-trained VAE .pt file (optional; trains if omitted)

SEED=${1:-42}
VAE_PATH=${2:-""}
=======
set -euo pipefail
# Run CMA-MAE on drug-like molecules (SmartCADD fitness).
# Usage: bash run_cma_mae_drug.sh [SEED]
#   SEED - random seed (default: 42)

SEED=${1:-42}
>>>>>>> Stashed changes

PYTHON=/home/dominic/miniconda3/envs/mol-evo/bin/python
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/cma_mae_drug_results_seed_${SEED}"

<<<<<<< Updated upstream
VAE_ARGS=""
if [ -n "$VAE_PATH" ]; then
    VAE_ARGS="--vae-path $VAE_PATH"
fi

=======
>>>>>>> Stashed changes
$PYTHON "$SCRIPT_DIR/main.py" \
    --seed "$SEED" \
    --atom-set drug \
    --fitness-mode smartcadd \
    --smartcadd-mode descriptors \
    --objective qed \
    --maximize \
<<<<<<< Updated upstream
    --latent-dim 64 \
    --vae-train-size 10000 \
    --vae-epochs 50 \
=======
>>>>>>> Stashed changes
    --learning-rate 0.01 \
    --n-emitters 5 \
    --cma-batch-size 36 \
    --sigma0 0.5 \
    --n_gen 500 \
    --pop_size 100 \
    --log_frequency 10 \
    --save_frequency 50 \
    --measure-keys num_atoms num_bonds \
    --archive-dims 10 10 \
    --output_dir "$OUTPUT_DIR" \
<<<<<<< Updated upstream
    $VAE_ARGS
=======
>>>>>>> Stashed changes
