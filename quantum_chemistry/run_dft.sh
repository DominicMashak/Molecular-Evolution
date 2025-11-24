python ~/Molecular-Evolution/quantum_chemistry/main.py \
    --calculator dft \
    --functional CAM-B3LYP \
    --basis "6-311G(d)" \
    --method full_tensor \
    --smiles "[O-][N+](=O)-C=C-C=C-C1=C-C=C(-N(-C)-C)-C=C1" \
    --solvent none \
    --field-strength 0.001 \
    --properties beta dipole homo_lumo_gap transition_dipole oscillator_strength gamma energy alpha