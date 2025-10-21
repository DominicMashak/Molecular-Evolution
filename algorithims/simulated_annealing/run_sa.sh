cd ~/Molecular-Evolution/algorithims/simulated_annealing
python simulated_annealing.py \
    --calculator semiempirical \
    --se-method PM7 \
    --initial_smiles "C1=CC=CC=C1" \
    --T_initial 100.0 \
    --T_min 0.01 \
    --cooling_rate 0.95 \
    --max_iterations 100 \
    --seed 0 \
    --cooling_schedule linear \
    --output_dir sa_results