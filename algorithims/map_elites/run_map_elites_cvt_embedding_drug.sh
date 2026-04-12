#!/bin/bash
<<<<<<< Updated upstream
=======
set -euo pipefail
>>>>>>> Stashed changes
# Run CVT-MAP-Elites with ChemBERTa-2 embeddings for drug-design optimization
# Uses transformer-based molecular embeddings as CVT behavior descriptors

# Random seed (change this for different runs)
SEED=${1:-42}  # Default to 42 if no argument provided

# SmartCADD mode: "descriptors" (fast, RDKit only) or "docking" (full pipeline with Smina)
MODE=${2:-descriptors}

# Objective to optimize in archive (qed, sa_score, docking_score, etc.)
OBJECTIVE=${3:-qed}

# Number of CVT centroids
N_CENTROIDS=${4:-100}

# Embedding dimensions (UMAP components)
EMB_DIMS=${5:-8}

echo "======================================================================"
echo "Running CVT-MAP-Elites (ChemBERTa embeddings) for Drug-Design Optimization"
echo "  Seed: ${SEED}"
echo "  Mode: ${MODE}"
echo "  Objective: ${OBJECTIVE}"
echo "  CVT centroids: ${N_CENTROIDS}"
echo "  Measures: embedding (ChemBERTa-2 MTR + UMAP)"
echo "  Embedding dims: ${EMB_DIMS}"
echo "  Device: auto-detect"
echo "======================================================================"

cd ~/Molecular-Evolution/algorithims/map_elites

# Set Python path
export PYTHONPATH="$HOME/Molecular-Evolution:$PYTHONPATH"

# Use mol-evo conda env python (needed for PyTorch/GPU support)
PYTHON="$HOME/miniconda3/envs/mol-evo/bin/python"

# Base command
CMD="$PYTHON main.py \
    --fitness-mode smartcadd \
    --smartcadd-mode ${MODE} \
    --atom-set drug \
    --objective-key ${OBJECTIVE} \
    --archive-type cvt \
    --n-centroids ${N_CENTROIDS} \
    --cvt-samples 50000 \
    --cvt-measures embedding \
    --embedding-model DeepChem/ChemBERTa-77M-MTR \
    --embedding-dims ${EMB_DIMS} \
    --embedding-device auto \
    --embedding-sample-size 10000 \
    --pop_size 50 \
    --n_gen 200 \
    --iterations_per_gen 50 \
    --log_frequency 5 \
    --save_frequency 10 \
    --output_dir cvt_emb_map_elites_drug_results_seed_${SEED}_${MODE}_${OBJECTIVE} \
    --seed ${SEED}"

# Add protein target if docking mode
if [ "$MODE" == "docking" ]; then
    PROTEIN_CODE=${6:-1AQ1}
    echo "  Protein: ${PROTEIN_CODE}"
    echo "======================================================================"
    CMD="$CMD --protein-code ${PROTEIN_CODE}"
else
    echo "======================================================================"
fi

# Execute
eval $CMD

echo ""
echo "======================================================================"
echo "Drug-design CVT-MAP-Elites (embedding) complete! Results saved to:"
echo "  cvt_emb_map_elites_drug_results_seed_${SEED}_${MODE}_${OBJECTIVE}/"
echo "======================================================================"
