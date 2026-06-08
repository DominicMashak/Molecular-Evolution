#!/usr/bin/env python3
import argparse
import sys
import os
import io
import numpy as np
import torch
from torch_geometric.data import Data
from contextlib import redirect_stdout

GPDRP_DIR = "/Users/rohanbasuroy/Documents/GitHub/GPDRP"
sys.path.insert(0, GPDRP_DIR)

from model.gin import GINConvNet

CELL_LINE  = "22RV1"
MODEL_PATH = os.path.join(GPDRP_DIR, "model.pth")


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


# ── cell line loader ──────────────────────────────────────────

def load_cell_feature(cell_line):
    original_dir = os.getcwd()
    os.chdir(GPDRP_DIR)

    try:
        from preprocess import save_cell_oge_matrix
        with redirect_stdout(io.StringIO()):  # suppress the "550" print
            cell_dict, cell_feature = save_cell_oge_matrix()
    finally:
        os.chdir(original_dir)

    if cell_line not in cell_dict:
        raise ValueError(f"Cell line '{cell_line}' not found")

    idx = cell_dict[cell_line]
    return cell_feature[idx].astype(np.float32)

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


# ── smiles → PyG Data object ──────────────────────────────────

def smiles_to_data(smiles, cell_feature):
    c_size, features, edge_index = smile_to_graph(smiles)

    x = torch.FloatTensor(np.array(features))

    if len(edge_index) > 0:
        edge_index_tensor = torch.LongTensor(edge_index).T
    else:
        edge_index_tensor = torch.LongTensor([[], []])

    data = Data(x=x, edge_index=edge_index_tensor)
    data.batch = torch.zeros(x.size(0), dtype=torch.long)
    data.target_ge = torch.FloatTensor(cell_feature).unsqueeze(0)  # [1, 1329]
    data.c_size = torch.LongTensor([c_size])                        # [1]

    return data


# ── main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles", required=True, help="SMILES string for the molecule")
    parser.add_argument("--cell-line", default=CELL_LINE, help="Cell line to use for prediction")
    parser.add_argument("--mode", default="single",
                    choices=["single", "average"],
                    help="single: predict for one cell line, average: predict across all 550")
    args = parser.parse_args()

    def inverse_transform(y):
        y = np.clip(y, 1e-6, 1 - 1e-6)
        return -10 * np.log(1 / y - 1)

    device = torch.device("cpu")

    # load model once regardless of mode
    model = GINConvNet().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    if args.mode == "average":
        all_cells = load_all_cell_features()
        preds = []
        for cell_name, cell_feature in all_cells.items():
            try:
                data = smiles_to_data(args.smiles, cell_feature).to(device)
                with torch.no_grad():
                    pred, _ = model(data)
                preds.append(inverse_transform(pred.item()))
            except Exception:
                pass
        avg = np.mean(preds) if preds else 0.0
        print(f"Predicted value (IC50): {avg:.4f}")
        return

    # single mode
    cell_feature = load_cell_feature(args.cell_line)
    data = smiles_to_data(args.smiles, cell_feature).to(device)

    with torch.no_grad():
        pred, _ = model(data)

    print(f"Predicted value (raw): {pred.item():.10f}")
    print(f"Predicted value (IC50): {inverse_transform(pred.item()):.4f}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)