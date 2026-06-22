#!/bin/bash
SEED=${1:-42}
cell_line=${2:-22RV1}
mu_value=${3:-10}
lambda_value=${4:-20}
qed_min=${5:-0.3}
filter_lipinski=${6:-true}
sa_max=${7:-6.0}
inference_mode=${8:-single}
initial_molecules_file=${9:-$HOME/Documents/GitHub/Molecular-Evolution/drug/initial_molecules.txt}
gpdrp_dir=${10:-$HOME/Documents/GitHub/GPDRP}

echo "======================================================================"
echo "Running (μ+λ) ES for GPDRP Optimization with seed ${SEED} and cell line ${cell_line}"
echo "======================================================================"

cd ~/Documents/GitHub/Molecular-Evolution/algorithims/mu_lambda

export PYTHONPATH="$HOME/Documents/GitHub/Molecular-Evolution:$PYTHONPATH"

python main.py \
    --fitness-mode gpdrp \
    --cell-line ${cell_line} \
    --objective lnic50 \
    --atom-set gpdrp \
    --minimize \
    --qed-min ${qed_min} \
    --sa-max ${sa_max} \
    --gpdrp-dir ${gpdrp_dir} \
    $( [ "$filter_lipinski" = "true" ] && echo "--filter-lipinski" ) \
    --mu ${mu_value} \
    --lambda ${lambda_value} \
    --n-gen 100 \
    --save-frequency 1 \
    --log-frequency 1 \
    --output-dir mu_lambda_gpdrp_results_seed_${SEED} \
    --seed ${SEED} \
    --initial-population-file ${initial_molecules_file} \
    --inference-mode ${inference_mode} \
    --verbose

echo ""
echo "======================================================================"
echo "GPDRP (μ+λ) optimization complete! Results saved to:"
echo "  mu_lambda_gpdrp_results_seed_${SEED}/"
echo "======================================================================"