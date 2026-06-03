#!/usr/bin/env python3
"""
molecule_profile.py

Molecular descriptor profile for a single molecule.

Usage:
    python molecule_profile.py --smiles "CCC1(O)C2=C(COC1=O)C(=O)N1CC3=CC4=CC=CC=C4N=C3C1=C2"
"""

import argparse
import sys
import os

MOL_EVO_DIR = "/Users/rohanbasuroy/Documents/GitHub/Molecular-Evolution"
sys.path.insert(0, MOL_EVO_DIR)

# load sascorer from whichever environment has it
_SASCORER_PATHS = [
    "/Users/rohanbasuroy/miniconda3/envs/mol-evo/share/RDKit/Contrib/SA_Score",
    "/Users/rohanbasuroy/miniconda3/envs/GPDRP/share/RDKit/Contrib/SA_Score",
]
for _path in _SASCORER_PATHS:
    if _path not in sys.path and os.path.isfile(os.path.join(_path, 'sascorer.py')):
        sys.path.insert(0, _path)
        break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles", required=True, help="SMILES string to profile")
    args = parser.parse_args()

    from rdkit import Chem
    from rdkit.Chem import Descriptors, QED
    from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds, CalcNumAromaticRings

    mol = Chem.MolFromSmiles(args.smiles)

    print("\n" + "="*55)
    print("MOLECULE PROFILE")
    print("="*55)
    print(f"SMILES: {args.smiles}")

    # ── structural ────────────────────────────────────────────
    print("\n── STRUCTURE ────────────────────────────────────────")
    if mol is None:
        print("  RDKit: INVALID — cannot parse SMILES")
        sys.exit(1)

    print(f"  RDKit valid:       YES")
    print(f"  Num atoms:         {mol.GetNumAtoms()}")
    print(f"  Num bonds:         {mol.GetNumBonds()}")
    print(f"  Num rings:         {mol.GetRingInfo().NumRings()}")
    print(f"  Aromatic rings:    {CalcNumAromaticRings(mol)}")
    print(f"  Rotatable bonds:   {CalcNumRotatableBonds(mol)}")

    has_triple  = any(b.GetBondType() == Chem.BondType.TRIPLE for b in mol.GetBonds())
    has_charges = any(a.GetFormalCharge() != 0 for a in mol.GetAtoms())
    atom_set    = {a.GetSymbol() for a in mol.GetAtoms()} - {'H'}
    print(f"  Atom types:        {sorted(atom_set)}")
    print(f"  Triple bonds:      {'yes' if has_triple else 'no'}")
    print(f"  Formal charges:    {'yes' if has_charges else 'no'}")

    # ── drug-likeness ─────────────────────────────────────────
    print("\n── DRUG-LIKENESS ────────────────────────────────────")
    qed  = QED.qed(mol)
    mw   = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd  = Descriptors.NumHDonors(mol)
    hba  = Descriptors.NumHAcceptors(mol)
    tpsa = Descriptors.TPSA(mol)

    violations = sum([mw > 500, logp > 5, hbd > 5, hba > 10])

    print(f"  QED:               {qed:.3f}   {'✓' if qed >= 0.3 else '✗ below 0.3'}")
    print(f"  MW:                {mw:.1f}  {'✓' if mw <= 500 else '✗ >500'}")
    print(f"  LogP:              {logp:.3f}  {'✓' if logp <= 5 else '✗ >5'}")
    print(f"  HBD:               {hbd}      {'✓' if hbd <= 5 else '✗ >5'}")
    print(f"  HBA:               {hba}      {'✓' if hba <= 10 else '✗ >10'}")
    print(f"  TPSA:              {tpsa:.1f}")
    print(f"  Lipinski violations: {violations}/4  {'✓' if violations <= 1 else '✗'}")

    # ── synthesizability ──────────────────────────────────────
    print("\n── SYNTHESIZABILITY ─────────────────────────────────")
    try:
        import sascorer
        sa = sascorer.calculateScore(mol)
        if sa <= 3:
            label = "easy"
        elif sa <= 6:
            label = "moderate"
        else:
            label = "hard"
        print(f"  SA score:          {sa:.3f}  ({label})")
    except Exception as e:
        print(f"  SA score:          unavailable ({e})")

    # ── validator ─────────────────────────────────────────────
    print("\n── EA VALIDATOR ─────────────────────────────────────")
    try:
        from molev_utils.molecule_generator import MoleculeGenerator
        g      = MoleculeGenerator(atom_set='gpdrp')
        passes = g.validate_as_smiles(args.smiles)
        print(f"  gpdrp validator (max_atoms=65): {'PASS' if passes else 'FAIL'}")
        if not passes:
            if mol.GetNumAtoms() > 65:
                print(f"  Reason: {mol.GetNumAtoms()} atoms exceeds limit of 65")
            if has_triple:
                print(f"  Reason: triple bonds not allowed")
            if has_charges:
                print(f"  Reason: formal charges not allowed")
            gpdrp_atomic = {1, 5, 6, 7, 8, 9, 16, 17, 35}
            bad = {a.GetAtomicNum() for a in mol.GetAtoms()} - gpdrp_atomic
            if bad:
                print(f"  Reason: exotic atomic numbers {bad}")
    except Exception as e:
        print(f"  Validator: unavailable ({e})")

    print("\n" + "="*55)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)