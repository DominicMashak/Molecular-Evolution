#!/bin/bash
<<<<<<< Updated upstream
=======
set -euo pipefail
>>>>>>> Stashed changes
# Run MOME (Multi-Objective MAP-Elites) for drug-design optimization
# Uses SmartCADD with multi-objective quality-diversity

# Random seed (change this for different runs)
SEED=${1:-42}  # Default to 42 if no argument provided

# SmartCADD mode: "descriptors" (fast, RDKit only) or "docking" (full pipeline with Smina)
MODE=${2:-descriptors}

echo "======================================================================"
echo "Running MOME for Drug-Design Optimization"
echo "  Seed: ${SEED}"
echo "  Mode: ${MODE}"
echo "======================================================================"

cd ~/Molecular-Evolution/algorithims/mome

# Set Python path
export PYTHONPATH="$HOME/Molecular-Evolution:$PYTHONPATH"

# Base command
CMD="python main.py \
    --fitness-mode smartcadd \
    --smartcadd-mode ${MODE} \
    --atom-set drug \
    --pop_size 30 \
    --n_gen 100 \
    --iterations_per_gen 30 \
    --log_frequency 5 \
    --save_frequency 10 \
    --output_dir mome_drug_results_seed_${SEED}_${MODE} \
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
echo "Drug-design MOME optimization complete! Results saved to:"
echo "  mome_drug_results_seed_${SEED}_${MODE}/"
echo "======================================================================"
