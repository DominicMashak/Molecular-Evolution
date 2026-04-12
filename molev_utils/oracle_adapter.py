"""
Universal OracleAdapter — domain-agnostic evaluation interface.

All oracle implementations for MO-CMA-MAE / CMA-MAE derive from OracleAdapter.
The single contract is:

    adapter.evaluate(smiles: str) -> Dict[str, Any]

The returned dict MUST contain:
  - 'error': None on success, or an error string on failure
  - Any objective / measure keys defined for the current domain

Concrete adapters wrap existing calculation backends (xtb-python, RDKit, SmartCADD).
New domains implement evaluate() without touching any optimizer code.

Typical usage (in main.py evaluate_solution closure):
    oracle = XTBNLOAdapter.from_args(args)
    props = oracle.evaluate(smiles)
    if props['error'] is not None:
        ...

Multi-fidelity (two-tier: xTB gating → HF evaluation):
    oracle = MultiFidelityAdapter(
        chain=[XTBNLOAdapter(...), HFNLOAdapter(...)],
        gate_key='homo_lumo_gap',
        gate_fraction=0.2,
    )
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np


# ── Base class ────────────────────────────────────────────────────────────────

class OracleAdapter(ABC):
    """
    Abstract base class for molecule evaluation oracles.

    Subclasses implement `evaluate(smiles)` for a specific domain and backend.
    The optimizer sees only this interface — switching domains means swapping
    adapter, nothing else.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short human-readable name for logging (e.g. 'XTB-NLO', 'RDKit-Drug')."""

    @abstractmethod
    def evaluate(self, smiles: str) -> Dict[str, Any]:
        """
        Evaluate a molecule given its SMILES string.

        Parameters
        ----------
        smiles : str
            SMILES string of the molecule to evaluate.

        Returns
        -------
        dict
            Must contain 'error' key:
              - 'error': None on success
              - 'error': str description on failure
            Plus all domain-specific property keys (objectives + measure proxies).
        """

    # ── Helpers available to all subclasses ───────────────────────────────────

    @staticmethod
    def error_result(message: str, extra_zeros: Optional[Dict[str, float]] = None
                     ) -> Dict[str, Any]:
        """Return a standard error dict, optionally with zeroed-out property keys."""
        result: Dict[str, Any] = {'error': message}
        if extra_zeros:
            result.update(extra_zeros)
        return result

    @property
    def fidelity_tiers(self) -> List[str]:
        """Tier names cheapest-first for multi-fidelity adapters.
        Single-fidelity adapters return a one-element list."""
        return [self.name]


# ── NLO domain: GFN2-xTB ─────────────────────────────────────────────────────

class XTBNLOAdapter(OracleAdapter):
    """
    NLO oracle: full-tensor β/γ/α via GFN2-xTB using xtb-python.

    Wraps QuantumChemistryInterface (quantum_chemistry/interface.py) configured
    for the xTB calculator.  Returns all 4 NLO objectives:
      - beta_gamma_ratio            (maximize)
      - total_energy_atom_ratio     (minimize)
      - alpha_range_distance        (minimize)
      - homo_lumo_gap_range_distance (minimize)

    Also exposes diagnostic properties: beta_mean, alpha_mean, homo_lumo_gap,
    total_energy, dipole_moment.
    """

    def __init__(
        self,
        xtb_method: str = 'GFN2-xTB',
        method: str = 'full_tensor',
        field_strength: float = 0.001,
        atom_set: str = 'nlo',
        verbose: bool = False,
    ):
        self.xtb_method = xtb_method
        self.method = method
        self.field_strength = field_strength
        self.atom_set = atom_set
        self.verbose = verbose
        self._interface = None   # lazy-loaded on first evaluate()

    @property
    def name(self) -> str:
        return f'XTB-NLO/{self.xtb_method}'

    def _get_interface(self):
        if self._interface is None:
            import sys, os
            _root = os.path.abspath(
                os.path.join(os.path.dirname(__file__), '..'))
            sys.path.insert(0, os.path.join(_root, 'quantum_chemistry'))
            from quantum_chemistry_interface import QuantumChemistryInterface
            self._interface = QuantumChemistryInterface(
                calculator='xtb',
                xtb_method=self.xtb_method,
                method=self.method,
                field_strength=self.field_strength,
                atom_set=self.atom_set,
                verbose=self.verbose,
            )
        return self._interface

    def evaluate(self, smiles: str) -> Dict[str, Any]:
        if smiles is None:
            return self.error_result('None SMILES')
        try:
            from rdkit import Chem
            if Chem.MolFromSmiles(smiles) is None:
                return self.error_result('Invalid SMILES')
        except Exception:
            pass

        try:
            result = self._get_interface().calculate(smiles)
        except Exception as e:
            return self.error_result(str(e))

        result.setdefault('error', None)
        return result

    @classmethod
    def from_args(cls, args) -> 'XTBNLOAdapter':
        """Build an XTBNLOAdapter from parsed argparse args."""
        return cls(
            xtb_method=getattr(args, 'xtb_method', 'GFN2-xTB'),
            method=getattr(args, 'method', 'full_tensor'),
            field_strength=getattr(args, 'field_strength', 0.001),
            atom_set=getattr(args, 'atom_set', 'nlo'),
            verbose=getattr(args, 'verbose', False),
        )


# ── Drug domain: RDKit descriptors ───────────────────────────────────────────

class RDKitDescriptorAdapter(OracleAdapter):
    """
    Drug design oracle: RDKit descriptors only.

    Returns QED, SA score (via sascorer), logP, MW, and structural features.
    No external tools required.  Fast (~1ms/mol).

    Objectives typically used:
      - qed      (maximize)
      - sa_score (minimize, lower = easier to synthesise)
    """

    @property
    def name(self) -> str:
        return 'RDKit-Drug'

    def evaluate(self, smiles: str) -> Dict[str, Any]:
        if smiles is None:
            return self.error_result('None SMILES')

        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors, QED
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return self.error_result('Invalid SMILES')

            qed_val = float(QED.qed(mol))
            mw = float(Descriptors.ExactMolWt(mol))
            logp = float(Descriptors.MolLogP(mol))
            hbd = int(Descriptors.NumHDonors(mol))
            hba = int(Descriptors.NumHAcceptors(mol))
            num_atoms = mol.GetNumAtoms()
            num_bonds = mol.GetNumBonds()

            # SA score via sascorer (part of RDKit Contrib)
            sa_score = self._sa_score(mol)

            return {
                'error': None,
                'qed': qed_val,
                'sa_score': sa_score,
                'logp': logp,
                'mw': mw,
                'hbd': float(hbd),
                'hba': float(hba),
                'num_atoms': float(num_atoms),
                'num_bonds': float(num_bonds),
            }
        except Exception as e:
            return self.error_result(str(e))

    @staticmethod
    def _sa_score(mol) -> float:
        """Compute SA score using RDKit sascorer (lower = easier to synthesise)."""
        try:
            from rdkit.Chem import RDConfig
            import sys, os
            sa_path = os.path.join(RDConfig.RDContribDir, 'SA_Score')
            if sa_path not in sys.path:
                sys.path.insert(0, sa_path)
            import sascorer
            return float(sascorer.calculateScore(mol))
        except Exception:
            # Fallback: approximate by ring + heavy atom complexity
            from rdkit.Chem import rdMolDescriptors
            rings = rdMolDescriptors.CalcNumRings(mol)
            return float(min(10.0, 1.0 + rings * 0.5 + mol.GetNumAtoms() * 0.02))


# ── Multi-fidelity adapter ─────────────────────────────────────────────────────

class MultiFidelityAdapter(OracleAdapter):
    """
    Composes two oracles into a cheap-then-expensive fidelity chain.

    Molecules are first evaluated with `cheap_oracle`.  Only those in the
    top `gate_fraction` by `gate_key` value are evaluated with `expensive_oracle`.
    The rest receive cheap-oracle values as their final result.

    This is an alternative to the argparse --fidelity-chain mechanism; it
    operates at the adapter level so any oracle combination can be composed
    without any changes to the optimizer.

    Note: This runs synchronously and is best suited for single-molecule
    evaluation.  For batched gating use the built-in FidelityChain in main.py.
    """

    def __init__(
        self,
        cheap: OracleAdapter,
        expensive: OracleAdapter,
        gate_key: str,
        gate_threshold: Optional[float] = None,
    ):
        """
        Parameters
        ----------
        cheap : OracleAdapter
            Fast oracle evaluated for every molecule.
        expensive : OracleAdapter
            Slow oracle evaluated only when cheap_result[gate_key] >= gate_threshold.
        gate_key : str
            Property key in the cheap result used for gating.
        gate_threshold : float, optional
            Gate value.  If None, every molecule is escalated (useful for testing).
        """
        self.cheap = cheap
        self.expensive = expensive
        self.gate_key = gate_key
        self.gate_threshold = gate_threshold

    @property
    def name(self) -> str:
        return f'MF({self.cheap.name}→{self.expensive.name})'

    @property
    def fidelity_tiers(self) -> List[str]:
        return [self.cheap.name, self.expensive.name]

    def evaluate(self, smiles: str) -> Dict[str, Any]:
        cheap_result = self.cheap.evaluate(smiles)
        if cheap_result.get('error') is not None:
            return cheap_result

        gate_val = cheap_result.get(self.gate_key)
        if gate_val is None or self.gate_threshold is None:
            # Escalate if gate key missing or no threshold set
            return self.expensive.evaluate(smiles)

        if float(gate_val) >= self.gate_threshold:
            return self.expensive.evaluate(smiles)

        # Below threshold: return cheap result with a tag indicating gating
        cheap_result['_fidelity_tier'] = self.cheap.name
        return cheap_result
