python ~/Molecular-Evolution/quantum_chemistry/main.py \
    --calculator semiempirical \
    --se-method MINDO3 \
    --method finite_field \
    --smiles "C1(-N)=C-C=C(-[N+](=O)-[O-])-C=C1" \
    --solvent none \
    --field-strength 0.001 \
    --properties beta dipole homo_lumo_gap transition_dipole oscillator_strength gamma energy alpha