#!/bin/bash
<<<<<<< Updated upstream
=======
set -euo pipefail
>>>>>>> Stashed changes
# Run NSGA-II for drug-design molecular optimization
# Uses SmartCADD for fast drug-likeness evaluation (QED, SA score, ADMET, etc.)

# Random seed (change this for different runs)
SEED=${1:-55}  # Default to 55 if no argument provided

# SmartCADD mode: "descriptors" (fast, RDKit only) or "docking" (full pipeline with Smina)
MODE=${2:-descriptors}

echo "======================================================================"
echo "Running NSGA-II for Drug-Design Optimization"
echo "  Seed: ${SEED}"
echo "  Mode: ${MODE}"
echo "======================================================================"

cd ~/Molecular-Evolution/algorithims/nsga2

# Set Python path
export PYTHONPATH="$HOME/Molecular-Evolution:$PYTHONPATH"

# Base command
CMD="python3 main.py \
    --fitness-mode smartcadd \
    --smartcadd-mode ${MODE} \
    --atom-set drug \
    --pop_size 50 \
    --n_gen 100 \
    --output_dir nsga2_drug_results_seed_${SEED}_${MODE} \
    --n-parents 25 \
    --n-children 25 \
    --no-stagnation-response \
    --seed ${SEED}"

# Different objectives based on mode
if [ "$MODE" == "docking" ]; then
    # Docking mode: requires protein target
    # Example: CDK2 (PDB: 6WPJ) - change as needed
    PROTEIN_CODE=${3:-6WPJ}

    echo "  Protein: ${PROTEIN_CODE}"
    echo "======================================================================"

    CMD="$CMD \
        --protein-code ${PROTEIN_CODE} \
        --objectives docking_score qed sa_score mol_weight_range_distance \
        --optimize minimize maximize minimize minimize \
        --reference-points 10.0 0.0 10.0 500.0"
else
    # Descriptors mode: fast, no docking (RECOMMENDED objectives)
    echo "======================================================================"

    CMD="$CMD \
        --objectives qed sa_score mol_weight_range_distance lipinski_violations \
        --optimize maximize minimize minimize minimize \
        --reference-points 0.0 10.0 500.0 5.0"
fi

# Execute
eval $CMD

echo ""
echo "======================================================================"
echo "Drug-design optimization complete! Results saved to:"
echo "  nsga2_drug_results_seed_${SEED}_${MODE}/"
echo "======================================================================"
