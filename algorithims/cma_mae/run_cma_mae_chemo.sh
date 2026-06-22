#!/bin/bash
SEED=${1:-42}
cell_line=${2:-22RV1}
qed_min=${3:-0.1}
filter_lipinski=${4:-true}
sa_max=${5:-6.0}
inference_mode=${6:-average}

echo "======================================================================"
echo "Running CMA-MAE for GPDRP with seed ${SEED} cell line ${cell_line}"
echo "======================================================================"

cd ~/Documents/GitHub/Molecular-Evolution/algorithims/cma_mae

export PYTHONPATH="$HOME/Documents/GitHub/Molecular-Evolution:$PYTHONPATH"

if [ "$filter_lipinski" = "true" ]; then
    LIPINSKI_FLAG="--filter-lipinski"
else
    LIPINSKI_FLAG=""
fi

python main.py \
    --seed ${SEED} \
    --fitness-mode gpdrp \
    --cell-line ${cell_line} \
    --inference-mode ${inference_mode} \
    --objective lnic50 \
    --minimize \
    --qed-min ${qed_min} \
    --sa-max ${sa_max} \
    ${LIPINSKI_FLAG} \
    --atom-set gpdrp \
    --encoding smiles \
    --latent-dim 64 \
    --vae-epochs 100 \
    --vae-hidden-dim 512 \
    --kl-anneal-epochs 100 \
    --n_gen 50 \
    --pop_size 50 \
    --cma-batch-size 36 \
    --n-emitters 5 \
    --measure-bounds 5 65 4 70 \
    --archive-dims 10 10 \
    --output_dir cma_mae_gpdrp_results_seed_${SEED} \
    --gpdrp-dir $HOME/Documents/GitHub/GPDRP \
    --verbose

echo "======================================================================"
echo "CMA-MAE complete! Results in cma_mae_gpdrp_results_seed_${SEED}/"
echo "======================================================================"