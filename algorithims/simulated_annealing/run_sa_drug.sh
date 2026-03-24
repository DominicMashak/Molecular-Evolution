#!/bin/bash
# Run Simulated Annealing for drug-design molecular optimization
# Uses SmartCADD for evaluation

# Random seed (change this for different runs)
SEED=${1:-42}  # Default to 42 if no argument provided

# SmartCADD mode: "descriptors" (fast, RDKit only) or "docking" (full pipeline with Smina)
MODE=${2:-descriptors}

# Objective to optimize (qed, sa_score, docking_score, etc.)
OBJECTIVE=${3:-qed}

echo "======================================================================"
echo "Running Simulated Annealing for Drug-Design Optimization"
echo "  Seed: ${SEED}"
echo "  Mode: ${MODE}"
echo "  Objective: ${OBJECTIVE}"
echo "======================================================================"

cd ~/Molecular-Evolution/algorithims/simulated_annealing

# Set Python path
export PYTHONPATH="$HOME/Molecular-Evolution:$PYTHONPATH"

# Base command
CMD="python simulated_annealing.py \
    --fitness-mode smartcadd \
    --smartcadd-mode ${MODE} \
    --atom-set drug \
    --objective ${OBJECTIVE} \
    --maximize \
    --T_initial 100.0 \
    --T_min 0.01 \
    --cooling_rate 0.95 \
    --max_iterations 200 \
    --cooling_schedule exponential \
    --output_dir sa_drug_results_seed_${SEED}_${MODE}_${OBJECTIVE} \
    --seed ${SEED}"

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
echo "Drug-design SA optimization complete! Results saved to:"
echo "  sa_drug_results_seed_${SEED}_${MODE}_${OBJECTIVE}/"
echo "======================================================================"
