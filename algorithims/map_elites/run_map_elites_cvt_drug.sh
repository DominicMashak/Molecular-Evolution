#!/bin/bash
# Run CVT-MAP-Elites (structural measures) for drug-design optimization
# Uses SmartCADD with quality-diversity and CVT archive

# Random seed (change this for different runs)
SEED=${1:-42}  # Default to 42 if no argument provided

# SmartCADD mode: "descriptors" (fast, RDKit only) or "docking" (full pipeline with Smina)
MODE=${2:-descriptors}

# Objective to optimize in archive (qed, sa_score, docking_score, etc.)
OBJECTIVE=${3:-qed}

# Number of CVT centroids
N_CENTROIDS=${4:-100}

echo "======================================================================"
echo "Running CVT-MAP-Elites for Drug-Design Optimization"
echo "  Seed: ${SEED}"
echo "  Mode: ${MODE}"
echo "  Objective: ${OBJECTIVE}"
echo "  CVT centroids: ${N_CENTROIDS}"
echo "  Measures: structural (num_atoms, num_bonds)"
echo "======================================================================"

cd ~/Molecular-Evolution/algorithims/map_elites

# Set Python path
export PYTHONPATH="$HOME/Molecular-Evolution:$PYTHONPATH"

# Base command
CMD="python main.py \
    --fitness-mode smartcadd \
    --smartcadd-mode ${MODE} \
    --atom-set drug \
    --objective-key ${OBJECTIVE} \
    --archive-type cvt \
    --n-centroids ${N_CENTROIDS} \
    --cvt-samples 50000 \
    --cvt-measures structural \
    --measure-bounds 5 50 4 60 \
    --pop_size 50 \
    --n_gen 200 \
    --iterations_per_gen 50 \
    --log_frequency 5 \
    --save_frequency 10 \
    --output_dir cvt_map_elites_drug_results_seed_${SEED}_${MODE}_${OBJECTIVE} \
    --seed ${SEED}"

# Add protein target if docking mode
if [ "$MODE" == "docking" ]; then
    PROTEIN_CODE=${5:-1AQ1}
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
echo "Drug-design CVT-MAP-Elites optimization complete! Results saved to:"
echo "  cvt_map_elites_drug_results_seed_${SEED}_${MODE}_${OBJECTIVE}/"
echo "======================================================================"
