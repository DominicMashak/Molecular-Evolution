#!/usr/bin/env python3
"""
ranking_comparison.py

Compares predicted vs actual drug rankings based on average LNIC50
across all cell lines. Rank 1 = lowest (most potent) LNIC50.

Usage:
    conda activate GPDRP
    python ranking_comparison.py
    python ranking_comparison.py --output ranking_results.csv --top 20
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
from scipy.stats import pearsonr, spearmanr

GPDRP_DIR    = "/Users/rohanbasuroy/Documents/GitHub/GPDRP_GDSC2"
sys.path.insert(0, GPDRP_DIR)

from model.gin import GINConvNet

MODEL_PATH     = os.path.join(GPDRP_DIR, "model.pth")
DRUGSMILE_PATH = os.path.join(GPDRP_DIR, "data/drugsmile_GDSC.csv")
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


def load_known_ic50():
    """Returns dict: drug_name -> list of known IC50 values across cell lines"""
    known = {}
    with open(DRUG_IC50_PATH, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            drug = row['Drug name'].strip()
            try:
                ic50 = float(row['IC50'])
                if drug not in known:
                    known[drug] = []
                known[drug].append(ic50)
            except ValueError:
                pass
    return known


def load_all_cell_features():
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
    data.batch     = torch.zeros(x.size(0), dtype=torch.long)
    data.target_ge = torch.FloatTensor(cell_feature).unsqueeze(0)
    data.c_size    = torch.LongTensor([c_size])
    return data


def inverse_transform(y):
    y = np.clip(y, 1e-6, 1 - 1e-6)
    return -10 * np.log(1 / y - 1)


def predict_drug_average(smiles, model, device, all_cells):
    """Predict average LNIC50 across all cell lines for one drug."""
    predictions = []
    for cell_feature in all_cells.values():
        try:
            data = smiles_to_data(smiles, cell_feature).to(device)
            with torch.no_grad():
                pred, _ = model(data)
            predictions.append(inverse_transform(pred.item()))
        except Exception:
            pass
    return np.mean(predictions) if predictions else None


# ── main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="ranking_comparison.csv",
                        help="Output CSV file")
    parser.add_argument("--top", type=int, default=None,
                        help="Only show top N drugs in printed table (default: all)")
    args = parser.parse_args()

    # load everything once
    print("Loading drug SMILES...")
    drug_smiles = load_drug_smiles()
    print(f"  {len(drug_smiles)} drugs")

    print("Loading known IC50 values...")
    known_ic50 = load_known_ic50()

    print("Loading model...")
    device = torch.device("cpu")
    model  = GINConvNet().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    print("Loading cell line features...")
    all_cells = load_all_cell_features()
    print(f"  {len(all_cells)} cell lines\n")

    # compute average IC50 per drug
    results = []
    total   = len(drug_smiles)

    for i, (drug_name, smiles) in enumerate(drug_smiles.items()):
        print(f"[{i+1:3d}/{total}] {drug_name}")

        # predicted average
        pred_avg = predict_drug_average(smiles, model, device, all_cells)

        # known average
        known_vals = known_ic50.get(drug_name)
        known_avg  = np.mean(known_vals) if known_vals else None

        if pred_avg is not None and known_avg is not None:
            results.append({
                'drug':      drug_name,
                'pred_avg':  pred_avg,
                'known_avg': known_avg,
                'n_known':   len(known_vals)
            })
            print(f"  predicted avg: {pred_avg:.4f}  known avg: {known_avg:.4f}")
        else:
            print(f"  skipped (missing data)")

    # rank by average LNIC50 — rank 1 = lowest = most potent
    results_sorted_pred  = sorted(results, key=lambda x: x['pred_avg'])
    results_sorted_known = sorted(results, key=lambda x: x['known_avg'])

    # assign ranks
    pred_rank  = {r['drug']: i+1 for i, r in enumerate(results_sorted_pred)}
    known_rank = {r['drug']: i+1 for i, r in enumerate(results_sorted_known)}

    for r in results:
        r['pred_rank']  = pred_rank[r['drug']]
        r['known_rank'] = known_rank[r['drug']]
        r['rank_diff']  = abs(r['pred_rank'] - r['known_rank'])

    # sort final table by predicted rank
    results.sort(key=lambda x: x['pred_rank'])

    # compute rank correlation
    pred_ranks  = np.array([r['pred_rank']  for r in results])
    known_ranks = np.array([r['known_rank'] for r in results])
    pred_avgs   = np.array([r['pred_avg']   for r in results])
    known_avgs  = np.array([r['known_avg']  for r in results])

    pearson_r,  pearson_p  = pearsonr(pred_avgs, known_avgs)
    spearman_r, spearman_p = spearmanr(pred_ranks, known_ranks)

    # print table
    n_show = args.top or len(results)
    print(f"\n{'='*75}")
    print(f"DRUG RANKING COMPARISON  (Rank 1 = most potent = lowest avg LNIC50)")
    print(f"{'='*75}")
    print(f"{'Drug Name':<25} {'Pred Avg':>10} {'Known Avg':>10} "
          f"{'Pred Rank':>10} {'Known Rank':>11} {'|Diff|':>8}")
    print(f"{'-'*75}")

    for r in results[:n_show]:
        print(f"{r['drug']:<25} {r['pred_avg']:>10.4f} {r['known_avg']:>10.4f} "
              f"{r['pred_rank']:>10d} {r['known_rank']:>11d} {r['rank_diff']:>8d}")

    print(f"\n{'='*75}")
    print(f"CORRELATION SUMMARY ({len(results)} drugs)")
    print(f"{'='*75}")
    print(f"  Pearson r  (avg LNIC50):   {pearson_r:.4f}  (p={pearson_p:.2e})")
    print(f"  Spearman r (rank order):   {spearman_r:.4f}  (p={spearman_p:.2e})")
    print(f"  Mean rank difference:      {np.mean([r['rank_diff'] for r in results]):.1f} positions")
    print(f"  Max rank difference:       {max(r['rank_diff'] for r in results)} positions")

    # save to CSV
    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'drug', 'pred_avg', 'known_avg', 'pred_rank', 'known_rank', 'rank_diff', 'n_known'
        ])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nFull results saved to {args.output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)