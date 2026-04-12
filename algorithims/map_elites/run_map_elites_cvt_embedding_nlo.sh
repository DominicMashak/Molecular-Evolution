#!/bin/bash
<<<<<<< Updated upstream
=======
set -euo pipefail
>>>>>>> Stashed changes
# Run MAP-Elites with CVT archive (ChemBERTa embedding measures) for NLO optimization
# Uses ChemBERTa-2 MTR molecular embeddings reduced by UMAP as CVT descriptors

# Random seed (change this for different runs)
SEED=${1:-42}  # Default to 42 if no argument provided

# Number of CVT centroids
N_CENTROIDS=${2:-100}

# Embedding dimensions (UMAP components)
EMB_DIMS=${3:-8}

echo "======================================================================"
echo "Running CVT-MAP-Elites (Embedding) for NLO Optimization"
echo "  Seed: ${SEED}"
echo "  CVT centroids: ${N_CENTROIDS}"
echo "  Measures: embedding (ChemBERTa-2 MTR + UMAP)"
echo "  Embedding dims: ${EMB_DIMS}"
echo "  Device: auto-detect"
echo "======================================================================"

cd ~/Molecular-Evolution/algorithims/map_elites

# Set Python path
export PYTHONPATH="$HOME/Molecular-Evolution:$PYTHONPATH"

# Use mol-evo conda env python
PYTHON="$HOME/miniconda3/envs/mol-evo/bin/python"

$PYTHON main.py \
    --fitness-mode qc \
    --calculator dft \
    --functional HF \
    --basis 3-21G \
    --method full_tensor \
    --field-strength 0.001 \
    --atom-set nlo \
    --objective-key beta_gamma_ratio \
    --archive-type cvt \
    --n-centroids ${N_CENTROIDS} \
    --cvt-samples 50000 \
    --cvt-measures embedding \
    --embedding-model DeepChem/ChemBERTa-77M-MTR \
    --embedding-dims ${EMB_DIMS} \
    --embedding-device auto \
    --embedding-sample-size 10000 \
    --pop_size 30 \
    --n_gen 20 \
    --iterations_per_gen 30 \
    --log_frequency 1 \
    --save_frequency 5 \
    --output_dir cvt_emb_map_elites_nlo_results_seed_${SEED} \
    --seed ${SEED}

echo ""
echo "======================================================================"
echo "CVT-Embedding MAP-Elites NLO optimization complete! Results saved to:"
echo "  cvt_emb_map_elites_nlo_results_seed_${SEED}/"
echo "======================================================================"
