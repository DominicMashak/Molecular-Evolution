#!/usr/bin/env python3
"""
correlation_analysis.py

Runs GPDRP inference across all 174 drugs and 550 cell lines,
then computes Pearson correlation between predicted and known LNIC50 values.

Usage:
    conda activate GPDRP
    python correlation_analysis.py
    python correlation_analysis.py --output results/correlation.csv
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
from scipy.stats import pearsonr

GPDRP_DIR  = "/Users/rohanbasuroy/Documents/GitHub/GPDRP_GDSC2"
sys.path.insert(0, GPDRP_DIR)

from model.gin import GINConvNet

MODEL_PATH       = os.path.join(GPDRP_DIR, "model.pth")
DRUGSMILE_PATH   = os.path.join(GPDRP_DIR, "data/drugsmile_GDSC.csv")
DRUG_IC50_PATH   = os.path.join(GPDRP_DIR, "data/drug_cl_ic.csv")


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

def load_drug_smiles():
    """Load drug name -> SMILES mapping from drugsmile_GDSC.csv"""
    drugs = {}
    with open(DRUGSMILE_PATH, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name   = row['name'].strip()
            smiles = row['CanonicalSMILES'].strip()
            if name and smiles:
                drugs[name] = smiles
    return drugs


def load_known_ic50():
    """
    Load all known IC50 values from drug_cl_ic.csv.
    Returns dict mapping (drug_name, cell_line) -> known_ic50
    """
    known = {}
    with open(DRUG_IC50_PATH, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            drug      = row['Drug name'].strip()
            cell_line = row['Cell line name'].strip()
            try:
                ic50  = float(row['IC50'])
                known[(drug, cell_line)] = ic50
            except ValueError:
                pass
    return known


def load_all_cell_features():
    """Load all cell line features from cell_ge.txt via GPDRP preprocessing."""
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


# ── inference ─────────────────────────────────────────────────

def smiles_to_data(smiles, cell_feature):
    c_size, features, edge_index = smile_to_graph(smiles)
    x = torch.FloatTensor(np.array(features))
    if len(edge_index) > 0:
        edge_index_tensor = torch.LongTensor(edge_index).T
    else:
        edge_index_tensor = torch.LongTensor([[], []])
    data = Data(x=x, edge_index=edge_index_tensor)
    data.batch      = torch.zeros(x.size(0), dtype=torch.long)
    data.target_ge  = torch.FloatTensor(cell_feature).unsqueeze(0)
    data.c_size     = torch.LongTensor([c_size])
    return data


def inverse_transform(y):
    y = np.clip(y, 1e-6, 1 - 1e-6)
    return -10 * np.log(1 / y - 1)


def predict_drug(smiles, model, device, all_cells):
    """
    Predict LNIC50 for one drug across all cell lines.
    Returns dict: cell_line -> predicted_lnic50
    """
    results = {}
    for cell_name, cell_feature in all_cells.items():
        try:
            data = smiles_to_data(smiles, cell_feature).to(device)
            with torch.no_grad():
                pred, _ = model(data)
            results[cell_name] = inverse_transform(pred.item())
        except Exception:
            pass
    return results


# ── main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="correlation_results.csv",
                        help="Output CSV file for per-drug/cell-line results")
    args = parser.parse_args()

    # load everything once
    print("Loading drug SMILES...")
    drug_smiles = load_drug_smiles()
    print(f"  {len(drug_smiles)} drugs found")

    print("Loading known IC50 values...")
    known_ic50 = load_known_ic50()
    print(f"  {len(known_ic50)} known drug-cell pairs found")

    print("Loading model...")
    device = torch.device("cpu")
    model  = GINConvNet().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    print("Loading cell line features...")
    all_cells = load_all_cell_features()
    print(f"  {len(all_cells)} cell lines found")

    # run inference across all drugs and cell lines
    all_predicted = []
    all_known     = []
    per_drug_r    = []
    per_cell_predictions = {}  # cell_line -> list of (predicted, known) pairs
    rows          = []         # for CSV output

    total_drugs = len(drug_smiles)
    print(f"\nRunning inference across {total_drugs} drugs × {len(all_cells)} cell lines...")
    print(f"Estimated time: ~{total_drugs * 6 / 60:.0f} minutes\n")

    for drug_idx, (drug_name, smiles) in enumerate(drug_smiles.items()):
        print(f"[{drug_idx+1:3d}/{total_drugs}] {drug_name}")

        try:
            predictions = predict_drug(smiles, model, device, all_cells)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        # collect paired (predicted, known) values for this drug
        drug_pred  = []
        drug_known = []

        for cell_line, pred_val in predictions.items():
            known_val = known_ic50.get((drug_name, cell_line))
            if known_val is None:
                continue

            drug_pred.append(pred_val)
            drug_known.append(known_val)
            all_predicted.append(pred_val)
            all_known.append(known_val)

            # accumulate per cell line
            if cell_line not in per_cell_predictions:
                per_cell_predictions[cell_line] = ([], [])
            per_cell_predictions[cell_line][0].append(pred_val)
            per_cell_predictions[cell_line][1].append(known_val)

            rows.append({
                'drug':      drug_name,
                'cell_line': cell_line,
                'predicted': pred_val,
                'known':     known_val,
                'diff':      pred_val - known_val
            })

        # per drug Pearson r
        if len(drug_pred) >= 2:
            r, p = pearsonr(drug_pred, drug_known)
            per_drug_r.append((drug_name, r, p, len(drug_pred)))
            print(f"  pairs={len(drug_pred)}  r={r:.4f}  p={p:.2e}")
        else:
            print(f"  insufficient pairs for correlation")

    # overall Pearson r
    all_predicted = np.array(all_predicted)
    all_known     = np.array(all_known)
    overall_r, overall_p = pearsonr(all_predicted, all_known)

    # per cell line Pearson r
    per_cell_r = []
    for cell_line, (preds, knowns) in per_cell_predictions.items():
        if len(preds) >= 2:
            r, p = pearsonr(preds, knowns)
            per_cell_r.append((cell_line, r, p, len(preds)))

    per_cell_r.sort(key=lambda x: x[1], reverse=True)
    per_drug_r.sort(key=lambda x: x[1], reverse=True)

    # print summary
    print("\n" + "="*60)
    print("CORRELATION ANALYSIS RESULTS")
    print("="*60)
    print(f"\nOverall ({len(all_predicted)} drug-cell pairs):")
    print(f"  Pearson r:  {overall_r:.4f}")
    print(f"  p-value:    {overall_p:.2e}")
    print(f"  MAE:        {np.mean(np.abs(all_predicted - all_known)):.4f}")

    print(f"\nTop 5 best predicted drugs (by per-drug r):")
    print(f"  {'Drug':<20} {'r':>8} {'p':>12} {'n':>6}")
    print(f"  {'-'*48}")
    for drug, r, p, n in per_drug_r[:5]:
        print(f"  {drug:<20} {r:>8.4f} {p:>12.2e} {n:>6}")

    print(f"\nTop 5 best predicted cell lines (by per-cell r):")
    print(f"  {'Cell line':<15} {'r':>8} {'p':>12} {'n':>6}")
    print(f"  {'-'*43}")
    for cell, r, p, n in per_cell_r[:5]:
        print(f"  {cell:<15} {r:>8.4f} {p:>12.2e} {n:>6}")

    # save full results to CSV
    import csv as csv_module
    with open(args.output, 'w', newline='') as f:
        writer = csv_module.DictWriter(f, fieldnames=['drug','cell_line','predicted','known','diff'])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nFull results saved to {args.output}")

    # save per drug correlation
    per_drug_file = args.output.replace('.csv', '_per_drug.csv')
    with open(per_drug_file, 'w', newline='') as f:
        writer = csv_module.writer(f)
        writer.writerow(['drug', 'pearson_r', 'p_value', 'n_pairs'])
        writer.writerows(per_drug_r)
    print(f"Per-drug correlation saved to {per_drug_file}")

    # save per cell line correlation
    per_cell_file = args.output.replace('.csv', '_per_cell.csv')
    with open(per_cell_file, 'w', newline='') as f:
        writer = csv_module.writer(f)
        writer.writerow(['cell_line', 'pearson_r', 'p_value', 'n_pairs'])
        writer.writerows(per_cell_r)
    print(f"Per-cell correlation saved to {per_cell_file}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)