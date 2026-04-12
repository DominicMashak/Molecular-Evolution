#!/bin/bash
<<<<<<< Updated upstream
=======
set -euo pipefail
>>>>>>> Stashed changes
# Run (μ+λ) Evolution Strategy for drug-design molecular optimization
# Uses SmartCADD with single-objective optimization

# Random seed (change this for different runs)
SEED=${1:-42}  # Default to 42 if no argument provided

# SmartCADD mode: "descriptors" (fast, RDKit only) or "docking" (full pipeline with Smina)
MODE=${2:-descriptors}

# Objective to optimize (qed, sa_score, docking_score, etc.)
OBJECTIVE=${3:-qed}

echo "======================================================================"
echo "Running (μ+λ) ES for Drug-Design Optimization"
echo "  Seed: ${SEED}"
echo "  Mode: ${MODE}"
echo "  Objective: ${OBJECTIVE}"
echo "======================================================================"

cd ~/Molecular-Evolution/algorithims/mu_lambda

# Set Python path
export PYTHONPATH="$HOME/Molecular-Evolution:$PYTHONPATH"

# Base command
CMD="python main.py \
    --fitness-mode smartcadd \
    --smartcadd-mode ${MODE} \
    --atom-set drug \
    --objective ${OBJECTIVE} \
    --maximize \
    --mu 20 \
    --lambda 40 \
    --n-gen 200 \
    --save-frequency 10 \
    --log-frequency 5 \
    --output-dir mu_lambda_drug_results_seed_${SEED}_${MODE}_${OBJECTIVE} \
    --seed ${SEED} \
    --verbose"

# Add protein target if docking mode
if [ "$MODE" == "docking" ]; then
    PROTEIN_CODE=${4:-1AQ1}
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
echo "Drug-design (μ+λ) optimization complete! Results saved to:"
echo "  mu_lambda_drug_results_seed_${SEED}_${MODE}_${OBJECTIVE}/"
echo "======================================================================"
