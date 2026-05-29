#!/bin/bash
SEED=${1:-42}
cell_line=${2:-22RV1}
mu_value=${3:-10}
lambda_value=${4:-20}
qed_min=${5:-0.3}
filter_lipinski=${6:-true}

echo "======================================================================"
echo "Running (μ+λ) ES for GPDRP Optimization with seed ${SEED} and cell line ${cell_line}"
echo "======================================================================"

cd ~/Documents/GitHub/Molecular-Evolution/algorithims/mu_lambda

export PYTHONPATH="$HOME/Documents/GitHub/Molecular-Evolution:$PYTHONPATH"

python main.py \
    --fitness-mode gpdrp \
    --cell-line ${cell_line} \
    --objective lnic50 \
    --atom-set drug \
    --minimize \
    --qed-min ${qed_min} \
    $( [ "$filter_lipinski" = "true" ] && echo "--filter-lipinski" ) \
    --mu ${mu_value} \
    --lambda ${lambda_value} \
    --n-gen 5 \
    --save-frequency 5 \
    --log-frequency 1 \
    --output-dir mu_lambda_gpdrp_results_seed_${SEED} \
    --seed ${SEED} \
    --verbose

echo ""
echo "======================================================================"
echo "GPDRP (μ+λ) optimization complete! Results saved to:"
echo "  mu_lambda_gpdrp_results_seed_${SEED}/"
echo "======================================================================"