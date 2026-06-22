#!/usr/bin/env python3
"""
inference_average.py

Runs GPDRP inference across cell lines for a given SMILES string.

Modes:
  all_cells       - inference on all 550 cell lines, known avg on available subset
  overlapping     - inference and known avg only on cell lines with known data
  unknown         - inference only on cell lines NOT in known data

Usage:
    conda activate GPDRP
    python inference_average.py --smiles "CC(=O)Oc1ccccc1C(=O)O" --drug-name "5-Fluorouracil" --mode all_cells
    python inference_average.py --smiles "CC(=O)Oc1ccccc1C(=O)O" --drug-name "5-Fluorouracil" --mode overlapping
    python inference_average.py --smiles "CC(=O)Oc1ccccc1C(=O)O" --drug-name "5-Fluorouracil" --mode unknown
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

GPDRP_DIR      = "/Users/rohanbasuroy/Documents/GitHub/GPDRP_GDSC2"
sys.path.insert(0, GPDRP_DIR)
sys.path.insert(0, "/Users/rohanbasuroy/Documents/GitHub/GPDRP")  # ← add this

from model.gin import GINConvNet

MODEL_PATH     = os.path.join(GPDRP_DIR, "model.pth")
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
    """Load all cell line features. Returns dict: cell_line -> np.array[1329]"""
    original_dir = os.getcwd()
    os.chdir(GPDRP_DIR)
    try:
        from preprocess import save_cell_oge_matrix
        with redirect_stdout(io.StringIO()):
            cell_dict, cell_feature = save_cell_oge_matrix()
    finally:
        os.chdir(original_dir)
    all_cells = {}
    for name, idx in cell_dict.items():
        all_cells[name] = cell_feature[idx].astype(np.float32)
    return all_cells


def load_known_ic50(drug_name):
    """Load known IC50 values for a drug. Returns dict: cell_line -> ic50"""
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
    data.batch     = torch.zeros(x.size(0), dtype=torch.long)
    data.target_ge = torch.FloatTensor(cell_feature).unsqueeze(0)
    data.c_size    = torch.LongTensor([c_size])
    return data


def inverse_transform(y):
    y = np.clip(y, 1e-6, 1 - 1e-6)
    return -10 * np.log(1 / y - 1)


def run_inference(smiles, model, device, cells_to_run):
    """
    Run inference on a specific subset of cell lines.
    Returns dict: cell_line -> predicted_lnic50
    """
    results = {}
    total   = len(cells_to_run)
    for i, (cell_name, cell_feature) in enumerate(cells_to_run.items()):
        print(f"\r  Progress: {i+1}/{total} cell lines", end='', flush=True)
        try:
            data = smiles_to_data(smiles, cell_feature).to(device)
            with torch.no_grad():
                pred, _ = model(data)
            results[cell_name] = inverse_transform(pred.item())
        except Exception:
            pass
    print()
    return results


def print_stats(label, values):
    """Print summary statistics for a set of LNIC50 values."""
    if not values:
        print(f"  {label}: no data")
        return
    arr = np.array(values)
    print(f"  {label}:")
    print(f"    n={len(arr)}  avg={np.mean(arr):.4f}  "
          f"std={np.std(arr):.4f}  "
          f"min={np.min(arr):.4f}  max={np.max(arr):.4f}")


# ── main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles",    required=True,
                        help="SMILES string to evaluate")
    parser.add_argument("--drug-name", default=None,
                        help="Drug name for known GDSC comparison")
    parser.add_argument("--mode",      default="all_cells",
                        choices=["all_cells", "overlapping", "unknown"],
                        help=(
                            "all_cells:   inference on all 550 cell lines, "
                            "known avg on available subset | "
                            "overlapping: inference and known only on shared cell lines | "
                            "unknown:     inference only on cell lines NOT in known data"
                        ))
    parser.add_argument("--cell-lines-file", default=None,
                    help="Path to file with cell line names to restrict inference to "
                         "(one per line). If not provided uses all available cell lines.")
    args = parser.parse_args()

    print(f"\nSMILES: {args.smiles}")
    print(f"Mode:   {args.mode}")
    print("="*60)

    # load model
    print("Loading model...")
    device = torch.device("cpu")
    model  = GINConvNet().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # load all cell features
    print("Loading cell line features...")
    all_cells = load_all_cell_features()
    print(f"  {len(all_cells)} total cell lines available")
    # filter to specific cell lines if provided
    if args.cell_lines_file:
        with open(args.cell_lines_file) as f:
            requested = set(l.strip() for l in f if l.strip() and not l.startswith('#'))
        all_cells = {c: v for c, v in all_cells.items() if c in requested}
        print(f"  Filtered to {len(all_cells)} cell lines from {args.cell_lines_file}")
    # load known IC50 values if drug name provided
    
    known = {}
    if args.drug_name:
        known = load_known_ic50(args.drug_name)
        print(f"  {len(known)} known IC50 values for '{args.drug_name}'")

    known_cell_lines      = set(known.keys())
    all_cell_lines        = set(all_cells.keys())
    overlapping_cells     = known_cell_lines & all_cell_lines
    unknown_cells         = all_cell_lines - known_cell_lines

    # ── determine which cell lines to run inference on ────────
    if args.mode == "all_cells":
        cells_to_infer = all_cells
        print(f"\nMode all_cells: running inference on all {len(cells_to_infer)} cell lines")

    elif args.mode == "overlapping":
        if not known:
            print("ERROR: --drug-name required for overlapping mode")
            sys.exit(1)
        cells_to_infer = {c: all_cells[c] for c in overlapping_cells if c in all_cells}
        print(f"\nMode overlapping: running inference on {len(cells_to_infer)} "
              f"shared cell lines ({len(known)} known, {len(all_cells)} total)")

    elif args.mode == "unknown":
        if not known:
            print("ERROR: --drug-name required for unknown mode")
            sys.exit(1)
        if len(unknown_cells) == 0:
            print(f"\nNOTE: '{args.drug_name}' covers all {len(all_cells)} cell lines "
                  f"— no unknown cell lines exist. Running inference on all {len(all_cells)}.")
            cells_to_infer = all_cells
        else:
            cells_to_infer = {c: all_cells[c] for c in unknown_cells}
            print(f"\nMode unknown: running inference on {len(cells_to_infer)} "
                  f"cell lines not in known data "
                  f"({len(known)} known, {len(unknown_cells)} unknown)")

    # ── run inference ─────────────────────────────────────────
    print("\nRunning inference...")
    predictions = run_inference(args.smiles, model, device, cells_to_infer)

    if not predictions:
        print("ERROR: No successful predictions")
        sys.exit(1)

    # ── print results ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")

    pred_values = list(predictions.values())
    best_cell   = min(predictions, key=predictions.get)
    worst_cell  = max(predictions, key=predictions.get)

    print_stats("Predicted LNIC50", pred_values)
    print(f"    most potent:  {best_cell}  ({predictions[best_cell]:.4f})")
    print(f"    least potent: {worst_cell}  ({predictions[worst_cell]:.4f})")

    # known average — always on whatever known data exists
    if known:
        known_values = list(known.values())
        print_stats("Known LNIC50   ", known_values)

        # per mode comparison
        if args.mode == "all_cells":
            # inference on all 550, known on subset
            overlap     = {c: (predictions[c], known[c])
                           for c in predictions if c in known}
            if overlap:
                pred_overlap  = [v[0] for v in overlap.values()]
                known_overlap = [v[1] for v in overlap.values()]
                mae = np.mean(np.abs(np.array(pred_overlap) - np.array(known_overlap)))
                print(f"\n  Overlap comparison ({len(overlap)} shared cell lines):")
                print(f"    Predicted avg (overlap): {np.mean(pred_overlap):.4f}")
                print(f"    Known avg    (overlap): {np.mean(known_overlap):.4f}")
                print(f"    MAE:                    {mae:.4f}")

        elif args.mode == "overlapping":
            # both on same cell lines
            overlap     = {c: (predictions[c], known[c])
                           for c in predictions if c in known}
            pred_vals   = np.array([v[0] for v in overlap.values()])
            known_vals  = np.array([v[1] for v in overlap.values()])
            mae         = np.mean(np.abs(pred_vals - known_vals))
            print(f"\n  Direct comparison on {len(overlap)} overlapping cell lines:")
            print(f"    Predicted avg: {np.mean(pred_vals):.4f}")
            print(f"    Known avg:     {np.mean(known_vals):.4f}")
            print(f"    MAE:           {mae:.4f}")

            # top 5
            print(f"\n  Top 5 most potent (predicted):")
            print(f"  {'Cell line':<15} {'Predicted':>10} {'Known':>10} {'Diff':>10}")
            print(f"  {'-'*47}")
            sorted_cells = sorted(overlap.items(), key=lambda x: x[1][0])
            for cell, (pred, kn) in sorted_cells[:5]:
                print(f"  {cell:<15} {pred:>10.4f} {kn:>10.4f} {pred-kn:>10.4f}")

        elif args.mode == "unknown":
            # inference on unknown cells, known on known cells — no direct comparison
            print(f"\n  NOTE: inference and known data are on different cell lines")
            print(f"  Predicted avg (unknown cells): {np.mean(pred_values):.4f}")
            print(f"  Known avg     (known cells):   {np.mean(known_values):.4f}")

    print("\n" + "="*60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)