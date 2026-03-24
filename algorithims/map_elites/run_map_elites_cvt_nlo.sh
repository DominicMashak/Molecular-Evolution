#!/bin/bash
# Run MAP-Elites with CVT archive (structural measures) for NLO optimization
# Uses continuous num_atoms/num_bonds as CVT descriptors instead of fixed grid bins

# Random seed (change this for different runs)
SEED=${1:-42}  # Default to 42 if no argument provided

# Number of CVT centroids
N_CENTROIDS=${2:-100}

echo "======================================================================"
echo "Running CVT-MAP-Elites for NLO Optimization"
echo "  Seed: ${SEED}"
echo "  CVT centroids: ${N_CENTROIDS}"
echo "  Measures: structural (num_atoms, num_bonds)"
echo "======================================================================"

cd ~/Molecular-Evolution/algorithims/map_elites

# Set Python path
export PYTHONPATH="$HOME/Molecular-Evolution:$PYTHONPATH"

python main.py \
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
    --cvt-measures structural \
    --measure-bounds 5 35 4 40 \
    --pop_size 30 \
    --n_gen 20 \
    --iterations_per_gen 30 \
    --log_frequency 1 \
    --save_frequency 5 \
    --output_dir cvt_map_elites_nlo_results_seed_${SEED} \
    --seed ${SEED}

echo ""
echo "======================================================================"
echo "CVT-MAP-Elites NLO optimization complete! Results saved to:"
echo "  cvt_map_elites_nlo_results_seed_${SEED}/"
echo "======================================================================"
