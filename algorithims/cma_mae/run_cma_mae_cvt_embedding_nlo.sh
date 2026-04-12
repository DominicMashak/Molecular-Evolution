#!/bin/bash
set -euo pipefail
# CVT-CMA-MAE with ChemBERTa-2 embedding measures for NLO optimization
SEED=${1:-42}

echo "======================================================================"
echo "Running CVT-CMA-MAE (ChemBERTa embeddings) for NLO with seed ${SEED}..."
echo "======================================================================"

cd ~/Molecular-Evolution/algorithims/cma_mae

export PYTHONPATH="$HOME/Molecular-Evolution/quantum_chemistry:$HOME/Molecular-Evolution/molev_utils:$PYTHONPATH"
PYTHON="$HOME/miniconda3/envs/mol-evo/bin/python"

$PYTHON main.py \
    --fitness-mode qc \
    --calculator dft \
    --functional HF \
    --basis 3-21G \
    --method full_tensor \
    --field-strength 0.001 \
    --atom-set nlo \
    --encoding smiles \
    --archive-type cvt \
    --n-centroids 100 \
    --cvt-measures embedding \
    --cvt-samples 50000 \
    --embedding-dims 10 \
    --embedding-sample-size 10000 \
    --embedding-device auto \
    --learning-rate 0.01 \
    --n-emitters 5 \
    --cma-batch-size 36 \
    --sigma0 0.5 \
    --objective beta_gamma_ratio \
    --maximize \
    --pop_size 100 \
    --n_gen 500 \
    --log_frequency 10 \
    --save_frequency 50 \
    --seed ${SEED} \
    --output_dir cma_mae_cvt_emb_nlo_results_seed_${SEED}

echo ""
echo "======================================================================"
echo "CVT-CMA-MAE (embedding) NLO complete! Results in:"
echo "  cma_mae_cvt_emb_nlo_results_seed_${SEED}/"
echo "======================================================================"
