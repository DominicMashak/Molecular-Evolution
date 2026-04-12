#!/bin/bash
<<<<<<< Updated upstream
=======
set -euo pipefail
>>>>>>> Stashed changes
# Run CVT-MOME (CVT Multi-Objective MAP-Elites) for NLO molecular optimization
# Uses Centroidal Voronoi Tessellation instead of fixed grid for niche definition

# Random seed (change this for different runs)
SEED=${1:-42}  # Default to 42 if no argument provided

echo "======================================================================"
echo "Running CVT-MOME for NLO Optimization with seed ${SEED}..."
echo "======================================================================"

cd ~/Molecular-Evolution/algorithims/mome

# Set Python path to include quantum_chemistry directory
export PYTHONPATH="$HOME/Molecular-Evolution/quantum_chemistry:$PYTHONPATH"

python main.py \
    --fitness-mode qc \
    --calculator dft \
    --functional HF \
    --basis 3-21G \
    --method full_tensor \
    --field-strength 0.001 \
    --atom-set nlo \
    --archive-type cvt \
    --n-centroids 100 \
    --cvt-samples 50000 \
    --measure-bounds 5 35 4 40 \
    --pop_size 20 \
    --n_gen 100 \
    --objectives beta_gamma_ratio total_energy_atom_ratio alpha_range_distance homo_lumo_gap_range_distance \
    --optimize maximize minimize minimize minimize \
    --iterations_per_gen 20 \
    --log_frequency 1 \
    --save_frequency 5 \
    --output_dir cvt_mome_nlo_results_seed_${SEED} \
    --seed ${SEED} \
    --reference-point 0.0 0.0 500.0 100.0

echo ""
echo "======================================================================"
echo "NLO CVT-MOME optimization complete! Results saved to:"
echo "  cvt_mome_nlo_results_seed_${SEED}/"
echo "======================================================================"
