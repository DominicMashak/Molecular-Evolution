#!/bin/bash
# Run CVT-MOME with ChemBERTa-2 embeddings for NLO molecular optimization
# Uses transformer-based molecular embeddings as CVT behavior descriptors

# Random seed (change this for different runs)
SEED=${1:-42}  # Default to 42 if no argument provided

echo "======================================================================"
echo "Running CVT-MOME (ChemBERTa embeddings) for NLO with seed ${SEED}..."
echo "======================================================================"

cd ~/Molecular-Evolution/algorithims/mome

# Set Python path
export PYTHONPATH="$HOME/Molecular-Evolution/quantum_chemistry:$HOME/Molecular-Evolution/molev_utils:$PYTHONPATH"

# Use mol-evo conda env python
PYTHON="$HOME/miniconda3/envs/mol-evo/bin/python"

$PYTHON main.py \
    --fitness-mode qc \
    --encoding smiles \
    --crossover-rate 0.0 \
    --calculator dft \
    --functional HF \
    --basis 3-21G \
    --method full_tensor \
    --field-strength 0.001 \
    --atom-set nlo \
    --archive-type cvt \
    --cvt-measures embedding \
    --embedding-device auto \
    --embedding-dims 10 \
    --embedding-sample-size 10000 \
    --n-centroids 100 \
    --cvt-samples 50000 \
    --pop_size 20 \
    --n_gen 100 \
    --objectives beta_gamma_ratio total_energy_atom_ratio alpha_range_distance homo_lumo_gap_range_distance \
    --optimize maximize minimize minimize minimize \
    --iterations_per_gen 20 \
    --log_frequency 1 \
    --save_frequency 5 \
    --output_dir cvt_emb_mome_nlo_results_seed_${SEED} \
    --seed ${SEED} \
    --reference-point 0.0 0.0 500.0 100.0

echo ""
echo "======================================================================"
echo "NLO CVT-MOME (embedding) complete! Results saved to:"
echo "  cvt_emb_mome_nlo_results_seed_${SEED}/"
echo "======================================================================"
