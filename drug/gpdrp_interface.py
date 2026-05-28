#!/usr/bin/env python3
"""
GPDRP interface for mu+lambda optimizer.
Mirrors the interface of smartcadd_interface.py and quantum_chemistry_interface.py.
"""

import subprocess
import os
from typing import Dict, Any

GPDRP_DIR = "/Users/rohanbasuroy/Documents/GitHub/GPDRP"
INFER_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "infer.py")
CONDA_PYTHON = "/Users/rohanbasuroy/miniconda3/envs/GPDRP/bin/python"


class GPDRPInterface:
    """
    Evaluation interface wrapping GPDRP drug response prediction.
    Called by MuLambdaOptimizer via eval_interface.calculate(smiles).
    """

    def __init__(self, cell_line: str = "22RV1", verbose: bool = False):
        self.cell_line = cell_line
        self.verbose = verbose

    def calculate(self, smiles: str) -> Dict[str, Any]:
        """
        Predict drug response for a SMILES string.
        Returns a dict matching the interface expected by MuLambdaOptimizer.

        Args:
            smiles: SMILES string of candidate molecule

        Returns:
            dict with 'lnic50' key and optional 'error' key
        """
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
                cwd=GPDRP_DIR  # needed for relative paths in preprocess.py
            )

            if proc.returncode != 0:
                return {"error": proc.stderr.strip()}

            # parse the two output lines
            lines = proc.stdout.strip().splitlines()
            lnic50 = None
            for line in lines:
                if line.startswith("Predicted value (IC50):"):
                    lnic50 = float(line.split(":")[1].strip())

            if lnic50 is None:
                return {"error": f"Could not parse output: {proc.stdout}"}

            if self.verbose:
                print(f"SMILES: {smiles} | LNIC50: {lnic50:.4f}")

            return {"lnic50": lnic50}

        except subprocess.TimeoutExpired:
            return {"error": "GPDRP inference timed out"}
        except Exception as e:
            return {"error": str(e)}