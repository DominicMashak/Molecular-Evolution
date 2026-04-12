"""
PMO (Practical Molecular Optimization) oracle adapter.

Wraps scoring functions into the evaluate_fn(smiles) -> dict interface used by
all optimizers in this codebase.  Provides PMO-standard budget tracking and
top-k AUC computation so results are directly comparable to published methods
(REINVENT, SMILES-GA, Graph-GA, MARS, DST, STONED, …).

References
----------
Gao, W. et al. "Sample efficiency matters: a benchmark for practical molecular
optimization." NeurIPS 2022. https://arxiv.org/abs/2206.12411

Oracles implemented locally (no external dependency beyond RDKit)
-----------------------------------------------------------------
  qed              Quantitative Estimate of Drug-likeness  (max, [0,1])
  penalized_logp   LogP – SA – ring-penalty                (max, unbounded)
  sa               Synthetic accessibility, inverted        (max, [0,1])
  logp             Wildman-Crippen LogP                     (max, unbounded)
  mw               Molecular weight (raw Daltons)           (max)
  ring_count       Number of rings in the molecule          (max)

TDC-backed oracles (require 'pip install PyTDC')
------------------------------------------------
  jnk3, gsk3b, drd2, celecoxib_rediscovery, troglitazone_rediscovery,
  thiothixene_rediscovery, albuterol_similarity, mestranol_similarity,
  isomers_c7h8n2o2, isomers_c9h10n2o2pf2cl, median1, median2,
  osimertinib_mpo, fexofenadine_mpo, ranolazine_mpo, perindopril_mpo,
  amlodipine_mpo, sitagliptin_mpo, zaleplon_mpo, valsartan_smarts,
  deco_hop, scaffold_hop

Usage
-----
    from molev_utils.pmo_oracle import create_oracle

    oracle = create_oracle("qed", budget=10_000)
    result = oracle.evaluate("c1ccccc1")
    # result: {'qed': 0.43, 'sa': 0.78, 'mol_weight': 78.0, ...}

    auc = oracle.get_auc_scores()
    # auc: {'top1_auc': 0.84, 'top10_auc': 0.81, 'top100_auc': 0.78,
    #       'final_top1': 0.91, 'final_top10': 0.88, 'final_top100': 0.84,
    #       'n_evaluations': 10000, 'n_valid': 9847}
"""

from __future__ import annotations

import os
import sys
import heapq
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# SA score loader (RDKit contrib)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Local oracle scoring functions  smiles -> float  (None on failure)
# ---------------------------------------------------------------------------

def _score_qed(smiles: str) -> Optional[float]:
    try:
        from rdkit import Chem
        from rdkit.Chem import QED
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return float(QED.qed(mol))
    except Exception:
        return None


def _score_sa(smiles: str) -> Optional[float]:
    """Return *inverted* SA score in [0, 1].  1 = trivially synthesisable."""
    try:
        from rdkit import Chem
        sascorer = _load_sascorer()
        if sascorer is None:
            return None
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        raw = sascorer.calculateScore(mol)  # 1 (easy) – 10 (hard)
        return float((10.0 - raw) / 9.0)   # invert and normalise to [0, 1]
    except Exception:
        return None


def _score_penalized_logp(smiles: str) -> Optional[float]:
    """
    Penalized LogP as used in most molecular-generation benchmarks.

    penalized_logP = logP(mol) - SA_raw(mol) - ring_penalty(mol)

    ring_penalty = max(0, largest_ring_size - 6)
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        sascorer = _load_sascorer()
        if sascorer is None:
            return None
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        logp = Descriptors.MolLogP(mol)
        sa = sascorer.calculateScore(mol)
        rings = mol.GetRingInfo().AtomRings()
        ring_penalty = max((max(len(r) for r in rings) - 6), 0) if rings else 0
        return float(logp - sa - ring_penalty)
    except Exception:
        return None


def _score_logp(smiles: str) -> Optional[float]:
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return float(Descriptors.MolLogP(mol))
    except Exception:
        return None


def _score_mw(smiles: str) -> Optional[float]:
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return float(Descriptors.MolWt(mol))
    except Exception:
        return None


def _score_ring_count(smiles: str) -> Optional[float]:
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return float(len(mol.GetRingInfo().AtomRings()))
    except Exception:
        return None


# Registry of locally-available oracles: name -> (scoring_fn, maximize)
_LOCAL_ORACLES: Dict[str, Tuple[Callable, bool]] = {
    'qed':            (_score_qed,            True),
    'penalized_logp': (_score_penalized_logp, True),
    'sa':             (_score_sa,             True),
    'logp':           (_score_logp,           True),
    'mw':             (_score_mw,             True),
    'ring_count':     (_score_ring_count,     True),
}

# ---------------------------------------------------------------------------
# TDC oracle wrapper (lazy import; graceful if TDC not installed)
# ---------------------------------------------------------------------------

_TDC_NAMES = {
    'jnk3', 'gsk3b', 'drd2',
    'celecoxib_rediscovery', 'troglitazone_rediscovery',
    'thiothixene_rediscovery',
    'albuterol_similarity', 'mestranol_similarity',
    'isomers_c7h8n2o2', 'isomers_c9h10n2o2pf2cl',
    'median1', 'median2',
    'osimertinib_mpo', 'fexofenadine_mpo', 'ranolazine_mpo',
    'perindopril_mpo', 'amlodipine_mpo', 'sitagliptin_mpo', 'zaleplon_mpo',
    'valsartan_smarts', 'deco_hop', 'scaffold_hop',
}


def _make_tdc_scorer(oracle_name: str) -> Callable:
    """Create a scoring function backed by TDC. Raises ImportError if TDC absent."""
    try:
        from tdc import Oracle as TDCOracle
    except ImportError:
        raise ImportError(
            f"TDC oracle '{oracle_name}' requires PyTDC: pip install PyTDC"
        )
    _oracle = TDCOracle(name=oracle_name)

    def _score(smiles: str) -> Optional[float]:
        try:
            v = _oracle(smiles)
            return float(v) if v is not None else None
        except Exception:
            return None

    return _score


# ---------------------------------------------------------------------------
# BudgetExhaustedError
# ---------------------------------------------------------------------------

class BudgetExhaustedError(RuntimeError):
    """Raised when PMOOracle.evaluate() is called beyond the allowed budget."""


# ---------------------------------------------------------------------------
# PMOOracle
# ---------------------------------------------------------------------------

class PMOOracle:
    """
    PMO-compatible oracle wrapper.

    Parameters
    ----------
    name :
        Oracle identifier (e.g. 'qed', 'jnk3').  Used as the primary score
        key in the returned properties dict.
    scoring_fn :
        Callable smiles -> float | None.  Must be thread-safe.
    budget :
        Maximum number of molecule evaluations.  Calls beyond this raise
        BudgetExhaustedError.
    maximize :
        True if higher score is better (default).  Stored for reference by
        downstream code; does not invert the raw score.
    aux_keys :
        Additional properties always computed and included in the returned
        dict (e.g. 'qed', 'sa', 'mol_weight').  These do *not* count toward
        the budget.
    checkpoint_interval :
        How often (in evaluations) to record a top-k snapshot for AUC
        computation.  Default 100.
    """

    def __init__(
        self,
        name: str,
        scoring_fn: Callable[[str], Optional[float]],
        budget: int = 10_000,
        maximize: bool = True,
        aux_keys: Optional[List[str]] = None,
        checkpoint_interval: int = 100,
    ):
        self.name = name
        self._scoring_fn = scoring_fn
        self.budget = budget
        self.maximize = maximize
        self.checkpoint_interval = checkpoint_interval

        # Auxiliary properties always added to result dict
        self._aux_scorers: Dict[str, Callable] = {}
        for key in (aux_keys or []):
            if key != name and key in _LOCAL_ORACLES:
                self._aux_scorers[key] = _LOCAL_ORACLES[key][0]

        # Budget tracking
        self.n_evaluations: int = 0
        self.n_valid: int = 0
        self._scores: List[float] = []       # all valid primary scores
        self._scored_smiles: List[Tuple[float, str]] = []  # (score, smiles) for diversity

        # AUC checkpoints: list of (eval_count, top1, top10, top100)
        self._checkpoints: List[Tuple[int, float, float, float]] = []
        self._next_checkpoint: int = checkpoint_interval

    # ------------------------------------------------------------------
    # Core evaluate interface
    # ------------------------------------------------------------------

    def evaluate(self, smiles: str) -> Dict[str, float]:
        """
        Evaluate a molecule and return a properties dict.

        The primary score is stored under key `self.name`.
        Additional standard properties (qed, sa, mol_weight) are always
        included for compatibility with archive measure computation.

        Raises BudgetExhaustedError if the evaluation budget is exhausted.
        """
        if self.n_evaluations >= self.budget:
            raise BudgetExhaustedError(
                f"Oracle '{self.name}' budget of {self.budget} evaluations exhausted."
            )

        self.n_evaluations += 1

        # Primary score
        score = self._scoring_fn(smiles)
        result: Dict[str, float] = {}

        if score is not None:
            result[self.name] = float(score)
            self._scores.append(float(score))
            self._scored_smiles.append((float(score), smiles))
            self.n_valid += 1
        else:
            # Invalid molecule — return sentinel 0.0 for archive compatibility
            result[self.name] = 0.0

        # Auxiliary properties
        for key, fn in self._aux_scorers.items():
            v = fn(smiles)
            result[key] = float(v) if v is not None else 0.0

        # Always include standard drug-likeness properties so archives can
        # use them as behavioral dimensions even if they are not the objective
        for key, fn in [
            ('qed', _score_qed),
            ('sa', _score_sa),
            ('mol_weight', _score_mw),
        ]:
            if key not in result:
                v = fn(smiles)
                result[key] = float(v) if v is not None else 0.0

        # Record AUC checkpoint
        if self.n_evaluations >= self._next_checkpoint:
            self._record_checkpoint()
            self._next_checkpoint += self.checkpoint_interval

        return result

    # ------------------------------------------------------------------
    # AUC / top-k statistics
    # ------------------------------------------------------------------

    def _topk_mean(self, k: int) -> float:
        if not self._scores:
            return 0.0
        top = heapq.nlargest(k, self._scores)
        return float(np.mean(top))

    def _record_checkpoint(self) -> None:
        self._checkpoints.append((
            self.n_evaluations,
            self._topk_mean(1),
            self._topk_mean(10),
            self._topk_mean(100),
        ))

    def get_auc_scores(self) -> Dict[str, float]:
        """
        Compute PMO-standard AUC metrics.

        Returns a dict with:
          top1_auc    — area under top-1 curve (normalised to [0, 1])
          top10_auc   — area under top-10 curve
          top100_auc  — area under top-100 curve
          final_top1  — best score seen
          final_top10 — mean of best 10 scores
          final_top100 — mean of best 100 scores
          n_evaluations — total calls to evaluate()
          n_valid     — calls that returned a non-None primary score
        """
        # Flush remaining evaluations into a final checkpoint
        if self._scores:
            self._record_checkpoint()

        def _auc(idx: int) -> float:
            if not self._checkpoints:
                return 0.0
            evals = [c[0] for c in self._checkpoints]
            vals  = [c[idx] for c in self._checkpoints]
            # Trapezoidal rule; normalise x-axis to [0, 1]
            total = float(self.budget)
            area = float(np.trapz(vals, x=[e / total for e in evals]))
            return float(np.clip(area, 0.0, 1.0))

        return {
            'top1_auc':       _auc(1),
            'top10_auc':      _auc(2),
            'top100_auc':     _auc(3),
            'final_top1':     self._topk_mean(1),
            'final_top10':    self._topk_mean(10),
            'final_top100':   self._topk_mean(100),
            'n_evaluations':  self.n_evaluations,
            'n_valid':        self.n_valid,
        }

    def save_results(self, output_dir: str) -> None:
        """Save AUC metrics and checkpoint history to output_dir."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        auc = self.get_auc_scores()
        with open(out / f'{self.name}_auc.json', 'w') as f:
            json.dump(auc, f, indent=2)

        rows = [{'eval': c[0], 'top1': c[1], 'top10': c[2], 'top100': c[3]}
                for c in self._checkpoints]
        with open(out / f'{self.name}_checkpoints.json', 'w') as f:
            json.dump(rows, f, indent=2)

        # Save top-1000 molecules by score (for diversity analysis)
        if self._scored_smiles:
            top = sorted(self._scored_smiles, key=lambda x: x[0], reverse=True)[:1000]
            top_rows = [{'score': s, 'smiles': m} for s, m in top]
            with open(out / f'{self.name}_top_molecules.json', 'w') as f:
                json.dump(top_rows, f, indent=2)

    def reset(self) -> None:
        """Reset budget and score tracking (keeps oracle function intact)."""
        self.n_evaluations = 0
        self.n_valid = 0
        self._scores = []
        self._scored_smiles = []
        self._checkpoints = []
        self._next_checkpoint = self.checkpoint_interval

    def is_exhausted(self) -> bool:
        return self.n_evaluations >= self.budget

    def remaining_budget(self) -> int:
        return max(0, self.budget - self.n_evaluations)

    def __repr__(self) -> str:
        return (
            f"PMOOracle(name={self.name!r}, budget={self.budget}, "
            f"used={self.n_evaluations}, valid={self.n_valid})"
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_oracle(
    name: str,
    budget: int = 10_000,
    checkpoint_interval: int = 100,
    aux_keys: Optional[List[str]] = None,
) -> PMOOracle:
    """
    Create a PMOOracle by name.

    Locally available (no extra deps):
        qed, penalized_logp, sa, logp, mw, ring_count

    TDC-backed (requires PyTDC):
        jnk3, gsk3b, drd2, celecoxib_rediscovery, troglitazone_rediscovery,
        thiothixene_rediscovery, albuterol_similarity, mestranol_similarity,
        isomers_c7h8n2o2, isomers_c9h10n2o2pf2cl, median1, median2,
        osimertinib_mpo, fexofenadine_mpo, ranolazine_mpo, perindopril_mpo,
        amlodipine_mpo, sitagliptin_mpo, zaleplon_mpo, valsartan_smarts,
        deco_hop, scaffold_hop

    Parameters
    ----------
    name :
        Oracle name (case-insensitive).
    budget :
        Maximum number of molecule evaluations (default 10 000).
    checkpoint_interval :
        Evaluation interval for AUC snapshots (default 100).
    aux_keys :
        Extra property keys to include in every evaluate() result.  Defaults
        to ['qed', 'sa', 'mol_weight'] (always included regardless).

    Returns
    -------
    PMOOracle ready to use as the evaluate_fn for any optimizer.
    """
    name_lower = name.lower()

    if name_lower in _LOCAL_ORACLES:
        fn, maximize = _LOCAL_ORACLES[name_lower]
        return PMOOracle(
            name=name_lower,
            scoring_fn=fn,
            budget=budget,
            maximize=maximize,
            aux_keys=aux_keys,
            checkpoint_interval=checkpoint_interval,
        )

    if name_lower in _TDC_NAMES:
        fn = _make_tdc_scorer(name_lower)
        return PMOOracle(
            name=name_lower,
            scoring_fn=fn,
            budget=budget,
            maximize=True,
            aux_keys=aux_keys,
            checkpoint_interval=checkpoint_interval,
        )

    raise ValueError(
        f"Unknown oracle '{name}'.  Available locally: "
        + ', '.join(sorted(_LOCAL_ORACLES))
        + ".  TDC-backed (requires PyTDC): "
        + ', '.join(sorted(_TDC_NAMES))
    )


def list_oracles() -> Dict[str, str]:
    """Return a dict mapping oracle names to their availability."""
    result = {k: 'local' for k in _LOCAL_ORACLES}
    try:
        import tdc  # noqa: F401
        for k in _TDC_NAMES:
            result[k] = 'tdc'
    except ImportError:
        for k in _TDC_NAMES:
            result[k] = 'tdc (not installed)'
    return result
