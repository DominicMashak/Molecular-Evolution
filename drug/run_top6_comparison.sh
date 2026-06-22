#!/bin/bash
cd /Users/rohanbasuroy/Documents/GitHub/Molecular-Evolution/drug

drugs=(
    "Bortezomib|B(C(CC(C)C)NC(=O)C(CC1=CC=CC=C1)NC(=O)C2=NC=CN=C2)(O)O"
    "Daporinad|C1CN(CCC1CCCCNC(=O)/C=C/C2=CN=CC=C2)C(=O)C3=CC=CC=C3"
    "Vinblastine|CCC1(CC2CC(C3=C(CCN(C2)C1)C4=CC=CC=C4N3)(C5=C(C=C6C(=C5)C78CCN9C7C(C=CC9)(C(C(C8N6C)(C(=O)OC)O)OC(=O)C)CC)OC)C(=O)OC)O"
    "Vinorelbine|CCC1=CC2CC(C3=C(CN(C2)C1)C4=CC=CC=C4N3)(C5=C(C=C6C(=C5)C78CCN9C7C(C=CC9)(C(C(C8N6C)(C(=O)OC)O)OC(=O)C)CC)OC)C(=O)OC"
    "Paclitaxel|CC1=C2C(C(=O)C3(C(CC4C(C3C(C(C2(C)C)(CC1OC(=O)C(C(C5=CC=CC=C5)NC(=O)C6=CC=CC=C6)O)O)OC(=O)C7=CC=CC=C7)(CO4)OC(=O)C)O)C)OC(=O)C"
    "Vincristine|CCC1(CC2CC(C3=C(CCN(C2)C1)C4=CC=CC=C4N3)(C5=C(C=C6C(=C5)C78CCN9C7C(C=CC9)(C(C(C8N6C=O)(C(=O)OC)O)OC(=O)C)CC)OC)C(=O)OC)O"
)

for entry in "${drugs[@]}"; do
    name="${entry%%|*}"
    smiles="${entry##*|}"
    echo "=============================="
    echo "Drug: $name"
    echo "=============================="
    python infer_average_comparison.py \
        --smiles "$smiles" \
        --drug-name "$name" \
        --mode all_cells \
        --cell-lines-file gdsc2_cell_lines.txt
    echo ""
done