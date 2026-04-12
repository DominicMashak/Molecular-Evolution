"""
drug_oracle.py — Pure-RDKit multi-objective drug design oracle.

Zero external dependencies beyond RDKit and numpy.  Designed for:
  1. Multi-objective ablation studies (DBSA vs quality-only vs no-surrogate).
  2. Direct REINVENT4 comparison via PMO-standard top-k AUC metrics.

Objectives
----------
qed                   QED drug-likeness score             maximize  [0, 1]
sa                    Inverted SA score (1=easy)           maximize  [0, 1]
logp_range_distance   Distance from Lipinski LogP [0, 5]  minimize  [0, ∞)
mol_weight_range_dist Distance from MW [150, 500] Da      minimize  [0, ∞)

Derived REINVENT4-comparison properties (also returned):
tpsa                  Topological polar surface area       reference
hbd                   H-bond donors                        reference
hba                   H-bond acceptors                     reference
lipinski_violations   Count of Lipinski Ro5 violations     reference

PMO-standard top-k AUC tracking
---------------------------------
DrugOracle tracks the top-1 / top-10 / top-100 QED scores across evaluations
so that results can be compared with the PMO benchmark leaderboard:

    Gao et al. "Sample efficiency matters: a benchmark for practical
    molecular optimization." NeurIPS 2022.

Results for REINVENT4 are available at:
    https://github.com/MolecularAI/REINVENT4

Usage
-----
    oracle = DrugOracle()                         # multi-objective
    props  = oracle.calculate("c1ccccc1")
    # props: {'qed': 0.43, 'sa': 0.78, ..., 'error': None}

    # PMO comparison: report top-k AUC at end of run
    auc = oracle.get_pmo_stats()
    # auc: {'top1_auc': 0.89, 'top10_auc': 0.84, ..., 'n_evaluations': 5000}
"""

from __future__ import annotations

import heapq
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── SA score loader (RDKit contrib) ──────────────────────────────────────────

_sa_module = None


def _load_sascorer():
    global _sa_module
    if _sa_module is not None:
        return _sa_module
    try:
        from rdkit.Chem import RDConfig
        sa_path = os.path.join(RDConfig.RDContribDir, 'SA_Score')
        if sa_path not in sys.path:
            sys.path.insert(0, sa_path)
        import sascorer
        _sa_module = sascorer
    except Exception:
        _sa_module = None
    return _sa_module


# ── Oracle class ─────────────────────────────────────────────────────────────

class DrugOracle:
    """
    Pure-RDKit multi-objective oracle for drug design.

    Computes 4 objectives per molecule (all from RDKit, ~1 ms/call):
        qed                   maximize
        sa                    maximize  (inverted SA, 1=easy)
        logp_range_distance   minimize
        mol_weight_range_dist minimize

    Also computes standard ADMET descriptors for behavioral CVT measures
    and for scaffold diversity analysis.

    Parameters
    ----------
    logp_range : tuple
        Optimal LogP window.  Default: (0.0, 5.0)  (Lipinski Ro5)
    mw_range : tuple
        Optimal MW window in Da.  Default: (150.0, 500.0)  (Lipinski Ro5)
    """

    # Objective keys in order — used by configs/run scripts
    OBJECTIVE_KEYS = [
        'qed',
        'sa',
        'logp_range_distance',
        'mol_weight_range_distance',
    ]
    OPTIMIZE_DIRECTIONS = ['maximize', 'maximize', 'minimize', 'minimize']
    # Worst-case reference point for hypervolume (all objectives at worst value)
    REFERENCE_POINT = [0.0, 0.0, 15.0, 500.0]

    def __init__(
        self,
        logp_range: Tuple[float, float] = (0.0, 5.0),
        mw_range: Tuple[float, float] = (150.0, 500.0),
    ):
        self._logp_lo, self._logp_hi = logp_range
        self._mw_lo, self._mw_hi = mw_range
        self.calculator_type = 'rdkit_drug'

        # PMO-style top-k tracking over QED (primary quality signal)
        self._n_evaluations: int = 0
        self._n_valid: int = 0
        self._qed_scores: List[float] = []
        self._checkpoints: List[Tuple[int, float, float, float]] = []  # (n, top1, top10, top100)
        self._checkpoint_interval: int = 100
        self._next_checkpoint: int = 100

    # ── Main entry point ─────────────────────────────────────────────────────

    def calculate(self, smiles: str, charge: int = 0, spin: int = 1) -> Dict[str, Any]:
        """
        Evaluate a molecule.  API-compatible with QuantumChemistryInterface.calculate().

        Returns a dict with keys:
          error, qed, sa, logp_range_distance, mol_weight_range_distance,
          tpsa, hbd, hba, logp, mol_weight, lipinski_violations, aromatic_rings
        """
        self._n_evaluations += 1

        result = self._zero_result()

        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors, QED
            from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds
            from rdkit.Chem.Lipinski import NumAromaticRings

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                result['error'] = 'Invalid SMILES'
                return result

            # Primary objectives
            result['qed'] = float(QED.qed(mol))
            sa_raw = self._sa_score(mol)
            result['sa'] = float((10.0 - sa_raw) / 9.0) if sa_raw is not None else 0.0

            logp = Descriptors.MolLogP(mol)
            mw   = Descriptors.MolWt(mol)
            tpsa = Descriptors.TPSA(mol)

            result['logp'] = float(logp)
            result['mol_weight'] = float(mw)
            result['tpsa'] = float(tpsa)

            # Range distance objectives (0 when inside optimal window)
            result['logp_range_distance'] = float(
                max(0.0, self._logp_lo - logp, logp - self._logp_hi)
            )
            result['mol_weight_range_distance'] = float(
                max(0.0, self._mw_lo - mw, mw - self._mw_hi)
            )

            # ADMET descriptors
            result['hbd'] = int(Descriptors.NumHDonors(mol))
            result['hba'] = int(Descriptors.NumHAcceptors(mol))
            result['rotatable_bonds'] = int(CalcNumRotatableBonds(mol))
            result['aromatic_rings'] = int(NumAromaticRings(mol))

            # Lipinski Ro5 violations (reference, not objective)
            violations = 0
            if mw > 500:   violations += 1
            if logp > 5:   violations += 1
            if result['hbd'] > 5:  violations += 1
            if result['hba'] > 10: violations += 1
            result['lipinski_violations'] = violations

            result['error'] = None
            self._n_valid += 1

            # PMO tracking (QED is the primary quality signal)
            self._qed_scores.append(result['qed'])
            if self._n_evaluations >= self._next_checkpoint:
                self._record_checkpoint()
                self._next_checkpoint += self._checkpoint_interval

        except Exception as exc:
            result['error'] = str(exc)

        return result

    # ── PMO-standard statistics ───────────────────────────────────────────────

    def get_pmo_stats(self) -> Dict[str, Any]:
        """
        Return PMO-standard top-k AUC metrics over QED.

        Returned keys match the PMO benchmark paper (Gao et al. NeurIPS 2022):
            top1_auc, top10_auc, top100_auc
            final_top1, final_top10, final_top100
            n_evaluations, n_valid
        """
        if self._qed_scores:
            self._record_checkpoint()

        def _topk(k: int) -> float:
            if not self._qed_scores:
                return 0.0
            return float(np.mean(heapq.nlargest(k, self._qed_scores)))

        def _auc(col: int) -> float:
            if not self._checkpoints:
                return 0.0
            total = max(self._n_evaluations, 1)
            xs = [c[0] / total for c in self._checkpoints]
            ys = [c[col] for c in self._checkpoints]
            return float(np.clip(np.trapz(ys, x=xs), 0.0, 1.0))

        return {
            'top1_auc':      _auc(1),
            'top10_auc':     _auc(2),
            'top100_auc':    _auc(3),
            'final_top1':    _topk(1),
            'final_top10':   _topk(10),
            'final_top100':  _topk(100),
            'n_evaluations': self._n_evaluations,
            'n_valid':       self._n_valid,
        }

    # ── Internals ─────────────────────────────────────────────────────────────

    def _sa_score(self, mol) -> Optional[float]:
        scorer = _load_sascorer()
        if scorer is None:
            return None
        try:
            return float(scorer.calculateScore(mol))
        except Exception:
            return None

    def _record_checkpoint(self) -> None:
        def _topk(k):
            return float(np.mean(heapq.nlargest(k, self._qed_scores))) if self._qed_scores else 0.0
        self._checkpoints.append((
            self._n_evaluations,
            _topk(1), _topk(10), _topk(100),
        ))

    def _zero_result(self) -> Dict[str, Any]:
        return {
            'error': None,
            'qed': 0.0,
            'sa':  0.0,
            'logp_range_distance': 15.0,
            'mol_weight_range_distance': 500.0,
            'logp': 0.0,
            'mol_weight': 0.0,
            'tpsa': 0.0,
            'hbd': 0,
            'hba': 0,
            'rotatable_bonds': 0,
            'aromatic_rings': 0,
            'lipinski_violations': 4,
            # QC-compatible keys (zero for drug mode)
            'beta_mean': 0.0,
            'beta_gamma_ratio': 0.0,
            'total_energy': 0.0,
            'homo_lumo_gap': 0.0,
            'alpha_mean': 0.0,
            'gamma': 0.0,
            'total_energy_atom_ratio': 0.0,
            'alpha_range_distance': 0.0,
            'homo_lumo_gap_range_distance': 0.0,
        }
