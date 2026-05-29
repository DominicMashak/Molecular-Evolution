#!/usr/bin/env python3
"""
inference_average.py

Runs GPDRP inference across all available cell lines for a given SMILES string.
Loads the model once and loops through all cell lines in the same process.

Usage:
    conda activate GPDRP
    python inference_average.py --smiles "CC(=O)Oc1ccccc1C(=O)O"
    python inference_average.py --smiles "CC(=O)Oc1ccccc1C(=O)O" --drug-name "5-Fluorouracil"
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

GPDRP_DIR = "/Users/rohanbasuroy/Documents/GitHub/GPDRP"
sys.path.insert(0, GPDRP_DIR)

from model.gin import GINConvNet

MODEL_PATH  = os.path.join(GPDRP_DIR, "model.pth")
CELL_GE_PATH = os.path.join(GPDRP_DIR, "data/cell_ge.txt")
DRUG_IC50_PATH = os.path.join(GPDRP_DIR, "data/drug_cl_ic.csv")


# ── atom featurization ────────────────────────────────────────

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception(f"input {x} not in allowable set {allowable_set}")
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


# ── data loading ──────────────────────────────────────────────

def load_all_cell_features():
    """
    Load gene expression features for all cell lines from cell_ge.txt.
    Returns dict mapping cell_line_name -> np.array of shape [1329]
    """
    original_dir = os.getcwd()
    os.chdir(GPDRP_DIR)

    try:
        from preprocess import save_cell_oge_matrix
        with redirect_stdout(io.StringIO()):
            cell_dict, cell_feature = save_cell_oge_matrix()
    finally:
        os.chdir(original_dir)

    # cell_dict maps name -> index, cell_feature is [550, 1329]
    all_cells = {}
    for name, idx in cell_dict.items():
        all_cells[name] = cell_feature[idx].astype(np.float32)

    return all_cells


def load_known_ic50(drug_name):
    """
    Load known IC50 values for a drug from drug_cl_ic.csv.
    Returns dict mapping cell_line_name -> known_ic50
    """
    known = {}
    with open(DRUG_IC50_PATH, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['Drug name'].strip() == drug_name:
                try:
                    known[row['Cell line name'].strip()] = float(row['IC50'])
                except ValueError:
                    pass
    return known


# ── inference ─────────────────────────────────────────────────

def smiles_to_data(smiles, cell_feature):
    c_size, features, edge_index = smile_to_graph(smiles)

    x = torch.FloatTensor(np.array(features))

    if len(edge_index) > 0:
        edge_index_tensor = torch.LongTensor(edge_index).T
    else:
        edge_index_tensor = torch.LongTensor([[], []])

    data = Data(x=x, edge_index=edge_index_tensor)
    data.batch = torch.zeros(x.size(0), dtype=torch.long)
    data.target_ge = torch.FloatTensor(cell_feature).unsqueeze(0)
    data.c_size = torch.LongTensor([c_size])

    return data


def inverse_transform(y):
    y = np.clip(y, 1e-6, 1 - 1e-6)
    return -10 * np.log(1 / y - 1)


def run_inference_all_cells(smiles, model, device, all_cells):
    """
    Run inference for a SMILES string across all cell lines.
    Returns dict mapping cell_line_name -> predicted_lnic50
    """
    results = {}
    total = len(all_cells)

    for i, (cell_name, cell_feature) in enumerate(all_cells.items()):
        print(f"\r  Progress: {i+1}/{total} cell lines", end='', flush=True)

        try:
            data = smiles_to_data(smiles, cell_feature).to(device)
            with torch.no_grad():
                pred, _ = model(data)
            lnic50 = inverse_transform(pred.item())
            results[cell_name] = lnic50
        except Exception as e:
            # skip cell lines that fail silently
            pass

    print()  # newline after progress
    return results


# ── main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles", required=True,
                        help="SMILES string to evaluate")
    parser.add_argument("--drug-name", default=None,
                        help="Optional drug name to compare against known GDSC values")
    args = parser.parse_args()

    print(f"\nSMILES: {args.smiles}")
    print("="*60)

    # load model once
    print("Loading model...")
    device = torch.device("cpu")
    model = GINConvNet().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # load all cell line features once
    print("Loading cell line features...")
    all_cells = load_all_cell_features()
    print(f"Found {len(all_cells)} cell lines")

    # run inference across all cell lines
    print("\nRunning inference across all cell lines...")
    results = run_inference_all_cells(args.smiles, model, device, all_cells)

    if not results:
        print("ERROR: No successful predictions")
        sys.exit(1)

    # compute statistics
    values      = np.array(list(results.values()))
    cell_names  = list(results.keys())

    avg_lnic50  = np.mean(values)
    std_lnic50  = np.std(values)
    min_lnic50  = np.min(values)
    max_lnic50  = np.max(values)
    best_cell   = cell_names[np.argmin(values)]   # lowest = most potent
    worst_cell  = cell_names[np.argmax(values)]   # highest = least potent

    print("\nResults:")
    print(f"  Average LNIC50:  {avg_lnic50:.4f}")
    print(f"  Std:             {std_lnic50:.4f}")
    print(f"  Min LNIC50:      {min_lnic50:.4f}  (cell line: {best_cell})  ← most potent")
    print(f"  Max LNIC50:      {max_lnic50:.4f}  (cell line: {worst_cell})  ← least potent")

    # compare against known values if drug name provided
    if args.drug_name:
        print(f"\nComparing against known GDSC values for '{args.drug_name}':")
        known = load_known_ic50(args.drug_name)

        if not known:
            print(f"  No known values found for '{args.drug_name}' in drug_cl_ic.csv")
        else:
            # find overlapping cell lines
            overlap = {c: (results[c], known[c]) for c in results if c in known}

            if not overlap:
                print("  No overlapping cell lines between predictions and known values")
            else:
                pred_vals  = np.array([v[0] for v in overlap.values()])
                known_vals = np.array([v[1] for v in overlap.values()])
                diff       = pred_vals - known_vals

                print(f"  Overlapping cell lines: {len(overlap)}")
                print(f"  Predicted avg:          {np.mean(pred_vals):.4f}")
                print(f"  Known avg:              {np.mean(known_vals):.4f}")
                print(f"  Mean absolute error:    {np.mean(np.abs(diff)):.4f}")
                print(f"  Max error:              {np.max(np.abs(diff)):.4f}")
                max_err_cell = max(overlap.items(), key=lambda x: abs(x[1][0] - x[1][1]))
                print(f"  Worst prediction: {max_err_cell[0]} predicted={max_err_cell[1][0]:.4f} known={max_err_cell[1][1]:.4f}")

                # show top 5 best predicted cell lines vs known
                print(f"\n  Top 5 most potent predictions vs known:")
                print(f"  {'Cell line':<15} {'Predicted':>10} {'Known':>10} {'Diff':>10}")
                print(f"  {'-'*47}")
                sorted_cells = sorted(overlap.items(), key=lambda x: x[1][0])
                for cell, (pred, known_val) in sorted_cells[:5]:
                    print(f"  {cell:<15} {pred:>10.4f} {known_val:>10.4f} {pred-known_val:>10.4f}")

    print("\n" + "="*60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)