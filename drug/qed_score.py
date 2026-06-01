#!/usr/bin/env python3
"""
Compute QED and Lipinski properties for one or more molecules.

Usage:
  python qed_score.py "CCO"
  python qed_score.py --file molecules.txt
  python qed_score.py --file molecules.csv   # must have a 'smiles' column
"""

import argparse
import sys


def compute_props(smiles: str):
    from rdkit import Chem
    from rdkit.Chem import Descriptors, QED

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol_weight = Descriptors.MolWt(mol)
    logp       = Descriptors.MolLogP(mol)
    hbd        = Descriptors.NumHDonors(mol)
    hba        = Descriptors.NumHAcceptors(mol)
    qed        = QED.qed(mol)

    violations = sum([
        mol_weight > 500,
        logp       > 5,
        hbd        > 5,
        hba        > 10,
    ])

    return {
        'qed':                 qed,
        'mol_weight':          mol_weight,
        'logp':                logp,
        'hbd':                 hbd,
        'hba':                 hba,
        'lipinski_violations': violations,
    }


def print_props(smiles: str, props: dict):
    print(f"SMILES : {smiles}")
    print(f"  QED                 : {props['qed']:.4f}")
    print(f"  Mol weight          : {props['mol_weight']:.2f}")
    print(f"  LogP                : {props['logp']:.2f}")
    print(f"  H-bond donors       : {props['hbd']}")
    print(f"  H-bond acceptors    : {props['hba']}")
    print(f"  Lipinski violations : {props['lipinski_violations']}")
    print()


def load_smiles_from_file(path: str):
    smiles_list = []
    if path.endswith('.csv'):
        import csv
        with open(path) as f:
            reader = csv.DictReader(f)
            col = next((c for c in reader.fieldnames if c.lower() == 'smiles'), None)
            if col is None:
                sys.exit("CSV has no 'smiles' column.")
            for row in reader:
                s = row[col].strip()
                if s:
                    smiles_list.append(s)
    else:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    smiles_list.append(line.split()[0])
    return smiles_list


def main():
    parser = argparse.ArgumentParser(description="Compute QED and Lipinski properties")
    parser.add_argument('smiles', nargs='?', help='SMILES string')
    parser.add_argument('--file', type=str, help='File with SMILES (TXT or CSV)')
    args = parser.parse_args()

    if not args.smiles and not args.file:
        parser.print_help()
        sys.exit(1)

    smiles_list = []
    if args.file:
        smiles_list = load_smiles_from_file(args.file)
    if args.smiles:
        smiles_list.insert(0, args.smiles)

    for smi in smiles_list:
        props = compute_props(smi)
        if props is None:
            print(f"SMILES : {smi}\n  ERROR: invalid SMILES\n")
        else:
            print_props(smi, props)


if __name__ == '__main__':
    main()
