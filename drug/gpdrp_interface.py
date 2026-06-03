#!/usr/bin/env python3
"""
GPDRP interface for mu+lambda optimizer.
Mirrors the interface of smartcadd_interface.py and quantum_chemistry_interface.py.
"""

import subprocess
import os
import sys
from typing import Dict, Any

# sascorer is an RDKit contrib script, not an installed package
_SASCORER_PATH = "/Users/rohanbasuroy/miniconda3/envs/mol-evo/share/RDKit/Contrib/SA_Score"
if _SASCORER_PATH not in sys.path and os.path.isfile(os.path.join(_SASCORER_PATH, 'sascorer.py')):
    sys.path.insert(0, _SASCORER_PATH)

GPDRP_DIR    = "/Users/rohanbasuroy/Documents/GitHub/GPDRP"
INFER_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "infer.py")
CONDA_PYTHON = "/Users/rohanbasuroy/miniconda3/envs/GPDRP/bin/python"

# ── drug-likeness filter thresholds ──────────────────────────
QED_MIN          = 0.3    # below this → not drug-like, reject
SA_MAX           = 6.0    # above this → too hard to synthesize, reject
MOL_WEIGHT_MAX   = 500    # Lipinski rule
LOGP_MAX         = 5      # Lipinski rule
HBD_MAX          = 5      # hydrogen bond donors
HBA_MAX          = 10     # hydrogen bond acceptors


class GPDRPInterface:
    """
    Evaluation interface wrapping GPDRP drug response prediction.
    Called by MuLambdaOptimizer via eval_interface.calculate(smiles).

    Filters molecules by QED and Lipinski rules before running GPDRP
    so invalid or non-drug-like molecules never reach the model.
    """

    def __init__(self, cell_line: str = "22RV1", verbose: bool = False,
                 qed_min: float = QED_MIN, filter_lipinski: bool = True,
                 sa_max: float = SA_MAX):
        self.cell_line       = cell_line
        self.verbose         = verbose
        self.qed_min         = qed_min
        self.filter_lipinski = filter_lipinski
        self.sa_max          = sa_max

    def _compute_rdkit_props(self, smiles: str) -> Dict[str, Any]:
        """
        Compute QED, Lipinski properties using RDKit.
        Returns dict with qed, mol_weight, logp, hbd, hba, lipinski_violations.
        Returns None if SMILES is invalid.
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors, QED
            import sascorer

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None

            mol_weight = Descriptors.MolWt(mol)
            logp       = Descriptors.MolLogP(mol)
            hbd        = Descriptors.NumHDonors(mol)
            hba        = Descriptors.NumHAcceptors(mol)
            qed        = QED.qed(mol)
            sa         = sascorer.calculateScore(mol)

            violations = sum([
                mol_weight > MOL_WEIGHT_MAX,
                logp       > LOGP_MAX,
                hbd        > HBD_MAX,
                hba        > HBA_MAX,
            ])

            return {
                'qed':                 qed,
                'sa_score':            sa,
                'mol_weight':          mol_weight,
                'logp':                logp,
                'hbd':                 hbd,
                'hba':                 hba,
                'lipinski_violations': violations,
            }
        except Exception as e:
            return None

    def calculate(self, smiles: str) -> Dict[str, Any]:
        """
        Predict drug response for a SMILES string.

        Pipeline:
          1. Compute RDKit properties (free, no subprocess)
          2. Filter by QED and Lipinski rules
          3. If passes → run GPDRP subprocess for LNIC50
          
        Returns dict with lnic50, qed, mol_weight, logp, lipinski_violations.
        Returns dict with 'error' key if molecule fails filter or inference.
        """

        # step 1 — compute RDKit properties
        props = self._compute_rdkit_props(smiles)

        if props is None:
            return {"error": f"Invalid SMILES: {smiles}"}

        # step 2 — apply filters
        if props['qed'] < self.qed_min:
            if self.verbose:
                print(f"REJECTED (QED={props['qed']:.3f} < {self.qed_min}): {smiles}")
            return {"error": f"QED too low: {props['qed']:.3f}"}

        if self.filter_lipinski and props['lipinski_violations'] > 2:
            if self.verbose:
                print(f"REJECTED (Lipinski violations={props['lipinski_violations']}): {smiles}")
            return {"error": f"Lipinski violations: {props['lipinski_violations']}"}

        if props['sa_score'] > self.sa_max:
            if self.verbose:
                print(f"REJECTED (SA={props['sa_score']:.3f} > {self.sa_max}): {smiles}")
            return {"error": f"SA score too high: {props['sa_score']:.3f}"}

        # step 3 — run GPDRP
        try:
            cmd = [
                CONDA_PYTHON,
                INFER_SCRIPT,
                "--smiles", smiles,
                "--cell-line", self.cell_line
            ]

            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
                cwd=GPDRP_DIR
            )

            if proc.returncode != 0:
                return {"error": proc.stderr.strip()}

            # parse output lines from infer.py
            lnic50 = None
            for line in proc.stdout.strip().splitlines():
                if line.startswith("Predicted value (IC50):"):
                    lnic50 = float(line.split(":")[1].strip())

            if lnic50 is None:
                return {"error": f"Could not parse output: {proc.stdout}"}

            if self.verbose:
                print(f"ACCEPTED  QED={props['qed']:.3f}  LNIC50={lnic50:.4f}  {smiles}")

            # return everything — optimizer uses lnic50, rest is logged
            return {
                "lnic50":                lnic50,
                "qed":                   props['qed'],
                "sa_score":              props['sa_score'],
                "mol_weight":            props['mol_weight'],
                "logp":                  props['logp'],
                "hbd":                   props['hbd'],
                "hba":                   props['hba'],
                "lipinski_violations":   props['lipinski_violations'],
            }

        except subprocess.TimeoutExpired:
            return {"error": "GPDRP inference timed out"}
        except Exception as e:
            return {"error": str(e)}