#!/usr/bin/env python3
"""
cell_line_sensitivity.py

For each cell line, counts how many times it is:
  - The MOST potent (lowest predicted LNIC50) across all drugs
  - The LEAST potent (highest predicted LNIC50) across all drugs

Ranks cell lines by each counter.
Cell lines with 0 for both are ignored.

Usage:
    conda activate GPDRP
    python cell_line_sensitivity.py
    python cell_line_sensitivity.py --output sensitivity_results.csv
"""

import argparse
import sys
import os
import io
import csv
import numpy as np
import torch
from torch_geometric.data import Data
from contextlib import redirect_stdout
from collections import defaultdict

GPDRP_DIR    = "/Users/rohanbasuroy/Documents/GitHub/GPDRP"
sys.path.insert(0, GPDRP_DIR)

from model.gin import GINConvNet

MODEL_PATH     = os.path.join(GPDRP_DIR, "model.pth")
DRUGSMILE_PATH = os.path.join(GPDRP_DIR, "data/drugsmile_GDSC.csv")


# ── atom featurization ────────────────────────────────────────

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception(f"input {x} not in allowable set")
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def atom_features(atom):
    return np.array(
        one_of_k_encoding_unk(atom.GetSymbol(),
            ['C','N','O','S','F','Si','P','Cl','Br','Mg','Na','Ca','Fe','As',
             'Al','I','B','V','K','Tl','Yb','Sb','Sn','Ag','Pd','Co','Se',
             'Ti','Zn','H','Li','Ge','Cu','Au','Ni','Cd','In','Mn','Zr','Cr',
             'Pt','Hg','Pb','Unknown']) +
        one_of_k_encoding(atom.GetDegree(), [0,1,2,3,4,5,6,7,8,9,10]) +
        one_of_k_encoding_unk(atom.GetTotalNumHs(), [0,1,2,3,4,5,6,7,8,9,10]) +
        one_of_k_encoding_unk(atom.GetImplicitValence(), [0,1,2,3,4,5,6,7,8,9,10]) +
        [atom.GetIsAromatic()]
    )

def smile_to_graph(smile):
    from rdkit import Chem
    import networkx as nx
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smile}")
    c_size = mol.GetNumAtoms()
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))
    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
    return c_size, features, edge_index

def smiles_to_data(smiles, cell_feature):
    c_size, features, edge_index = smile_to_graph(smiles)
    x = torch.FloatTensor(np.array(features))
    if len(edge_index) > 0:
        edge_index_tensor = torch.LongTensor(edge_index).T
    else:
        edge_index_tensor = torch.LongTensor([[], []])
    data = Data(x=x, edge_index=edge_index_tensor)
    data.batch     = torch.zeros(x.size(0), dtype=torch.long)
    data.target_ge = torch.FloatTensor(cell_feature).unsqueeze(0)
    data.c_size    = torch.LongTensor([c_size])
    return data

def inverse_transform(y):
    y = np.clip(y, 1e-6, 1 - 1e-6)
    return -10 * np.log(1 / y - 1)


# ── data loading ──────────────────────────────────────────────

def load_drug_smiles():
    drugs = {}
    with open(DRUGSMILE_PATH, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name   = row['name'].strip()
            smiles = row['CanonicalSMILES'].strip()
            if name and smiles:
                drugs[name] = smiles
    return drugs

def load_all_cell_features():
    original_dir = os.getcwd()
    os.chdir(GPDRP_DIR)
    try:
        from preprocess import save_cell_oge_matrix
        with redirect_stdout(io.StringIO()):
            cell_dict, cell_feature = save_cell_oge_matrix()
    finally:
        os.chdir(original_dir)
    return {name: cell_feature[idx].astype(np.float32)
            for name, idx in cell_dict.items()}


# ── main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="cell_line_sensitivity.csv",
                        help="Output CSV file")
    args = parser.parse_args()

    print("Loading drug SMILES...")
    drug_smiles = load_drug_smiles()
    print(f"  {len(drug_smiles)} drugs")

    print("Loading model...")
    device = torch.device("cpu")
    model  = GINConvNet().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    print("Loading cell line features...")
    all_cells = load_all_cell_features()
    print(f"  {len(all_cells)} cell lines\n")

    # counters
    most_potent_count  = defaultdict(int)   # cell line → times it was most potent
    least_potent_count = defaultdict(int)   # cell line → times it was least potent

    total = len(drug_smiles)

    for i, (drug_name, smiles) in enumerate(drug_smiles.items()):
        print(f"[{i+1:3d}/{total}] {drug_name}")

        # predict for all cell lines
        predictions = {}
        for cell_name, cell_feature in all_cells.items():
            try:
                data = smiles_to_data(smiles, cell_feature).to(device)
                with torch.no_grad():
                    pred, _ = model(data)
                predictions[cell_name] = inverse_transform(pred.item())
            except Exception:
                pass

        if not predictions:
            continue

        # find most and least potent cell line for this drug
        best_cell  = min(predictions, key=predictions.get)  # lowest = most potent
        worst_cell = max(predictions, key=predictions.get)  # highest = least potent

        most_potent_count[best_cell]   += 1
        least_potent_count[worst_cell] += 1

        print(f"  most potent:  {best_cell} ({predictions[best_cell]:.4f})")
        print(f"  least potent: {worst_cell} ({predictions[worst_cell]:.4f})")

    # combine results — ignore cell lines with 0 for both
    all_cell_lines = set(most_potent_count.keys()) | set(least_potent_count.keys())

    results = []
    for cell in all_cell_lines:
        most  = most_potent_count[cell]
        least = least_potent_count[cell]
        if most == 0 and least == 0:
            continue
        results.append({
            'cell_line':         cell,
            'most_potent_count': most,
            'least_potent_count': least,
            'net_sensitivity':   most - least   # positive = generally sensitive
        })

    # sort by most potent count descending
    results_by_most  = sorted(results, key=lambda x: x['most_potent_count'],  reverse=True)
    results_by_least = sorted(results, key=lambda x: x['least_potent_count'], reverse=True)

    # assign ranks
    for i, r in enumerate(results_by_most):
        r['rank_most_potent'] = i + 1
    for i, r in enumerate(results_by_least):
        r['rank_least_potent'] = i + 1

    # print summary
    print(f"\n{'='*65}")
    print("CELL LINE SENSITIVITY SUMMARY")
    print(f"{'='*65}")

    print(f"\nTop 15 most frequently MOST POTENT cell lines:")
    print(f"{'Cell line':<20} {'Most potent':>12} {'Least potent':>13} {'Net':>6}")
    print(f"{'-'*53}")
    for r in results_by_most[:15]:
        print(f"{r['cell_line']:<20} {r['most_potent_count']:>12} "
              f"{r['least_potent_count']:>13} {r['net_sensitivity']:>6}")

    print(f"\nTop 15 most frequently LEAST POTENT cell lines:")
    print(f"{'Cell line':<20} {'Most potent':>12} {'Least potent':>13} {'Net':>6}")
    print(f"{'-'*53}")
    for r in results_by_least[:15]:
        print(f"{r['cell_line']:<20} {r['most_potent_count']:>12} "
              f"{r['least_potent_count']:>13} {r['net_sensitivity']:>6}")

    # save full results sorted by most potent rank
    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'cell_line', 'most_potent_count', 'least_potent_count',
            'net_sensitivity', 'rank_most_potent', 'rank_least_potent'
        ])
        writer.writeheader()
        writer.writerows(results_by_most)

    print(f"\nFull results saved to {args.output}")
    print(f"Total cell lines with non-zero counts: {len(results)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)