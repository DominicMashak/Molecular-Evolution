python ~/Molecular-Evolution/quantum_chemistry/main.py \
    --calculator dft \
    --functional B3LYP \
    --basis 6-31G \
    --method full_tensor \
    --smiles "CN(NC=N)c1cn(C)c(=O)n(N)c1=N" \
    --solvent none \
    --field-strength 0.001 \
    --properties beta dipole homo_lumo_gap transition_dipole oscillator_strength gamma energy alpha