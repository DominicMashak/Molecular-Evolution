#!/usr/bin/env python3
"""
Compute the Synthetic Accessibility (SA) score for one or more molecules.
Score ranges from 1 (easy to synthesize) to 10 (hard to synthesize).

Usage:
  python sa_score.py "CCO"
  python sa_score.py --file molecules.txt
  python sa_score.py --file molecules.csv   # must have a 'smiles' column
"""

import argparse
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'molev_utils')))

# sascorer is an RDKit contrib script, not an installed package
_SASCORER_PATHS = [
    "/Users/rohanbasuroy/miniconda3/envs/GPDRP/share/RDKit/Contrib/SA_Score",
    os.path.join(os.path.dirname(__file__), '..', 'molev_utils'),
]
for _p in _SASCORER_PATHS:
    if os.path.isfile(os.path.join(_p, 'sascorer.py')):
        sys.path.insert(0, _p)
        break


def compute_sa(smiles: str):
    from rdkit import Chem
    import sascorer

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return sascorer.calculateScore(mol)


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
    parser = argparse.ArgumentParser(description="Compute SA score (1=easy, 10=hard to synthesize)")
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
        score = compute_sa(smi)
        if score is None:
            print(f"SMILES : {smi}\n  ERROR: invalid SMILES\n")
        else:
            print(f"SMILES : {smi}")
            print(f"  SA score : {score:.4f}  (1=easy, 10=hard)")
            print()


if __name__ == '__main__':
    main()
