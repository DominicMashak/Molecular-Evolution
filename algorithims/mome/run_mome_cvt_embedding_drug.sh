#!/bin/bash
# Run CVT-MOME with ChemBERTa-2 embeddings for drug-design optimization
# Uses transformer-based molecular embeddings as CVT behavior descriptors

# Random seed (change this for different runs)
SEED=${1:-42}  # Default to 42 if no argument provided

# SmartCADD mode: "descriptors" (fast, RDKit only) or "docking" (full pipeline with Smina)
MODE=${2:-descriptors}

echo "======================================================================"
echo "Running CVT-MOME (ChemBERTa embeddings) for Drug-Design Optimization"
echo "  Seed: ${SEED}"
echo "  Mode: ${MODE}"
echo "  Measures: embedding (ChemBERTa-2 MTR + UMAP)"
echo "======================================================================"

cd ~/Molecular-Evolution/algorithims/mome

# Set Python path
export PYTHONPATH="$HOME/Molecular-Evolution/molev_utils:$HOME/Molecular-Evolution:$PYTHONPATH"

# Use mol-evo conda env python (needed for PyTorch/GPU support)
PYTHON="$HOME/miniconda3/envs/mol-evo/bin/python"

# Base command
CMD="$PYTHON main.py \
    --fitness-mode smartcadd \
    --smartcadd-mode ${MODE} \
    --atom-set drug \
    --archive-type cvt \
    --cvt-measures embedding \
    --embedding-model DeepChem/ChemBERTa-77M-MTR \
    --embedding-device auto \
    --embedding-dims 8 \
    --embedding-sample-size 10000 \
    --n-centroids 100 \
    --cvt-samples 50000 \
    --pop_size 30 \
    --n_gen 100 \
    --iterations_per_gen 30 \
    --log_frequency 5 \
    --save_frequency 10 \
    --output_dir cvt_emb_mome_drug_results_seed_${SEED}_${MODE} \
    --seed ${SEED}"

# Different objectives based on mode
if [ "$MODE" == "docking" ]; then
    # Docking mode: requires protein target
    PROTEIN_CODE=${3:-1AQ1}

    echo "  Protein: ${PROTEIN_CODE}"
    echo "======================================================================"

    CMD="$CMD \
        --protein-code ${PROTEIN_CODE} \
        --objectives docking_score qed sa_score lipinski_violations \
        --optimize minimize maximize minimize minimize \
        --reference-point 10.0 0.0 10.0 5.0"
else
    # Descriptors mode: fast, no docking
    echo "======================================================================"

    CMD="$CMD \
        --objectives qed sa_score lipinski_violations mol_weight \
        --optimize maximize minimize minimize minimize \
        --reference-point 0.0 10.0 5.0 1000.0"
fi

# Execute
eval $CMD

echo ""
echo "======================================================================"
echo "Drug-design CVT-MOME (embedding) complete! Results saved to:"
echo "  cvt_emb_mome_drug_results_seed_${SEED}_${MODE}/"
echo "======================================================================"
