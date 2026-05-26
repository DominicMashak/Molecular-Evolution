"""
molopt_bridge.py
================
Bridge that lets mol_opt algorithm implementations (graph_ga, smiles_ga, reinvent)
run against our PMOOracle without modifications to their code.

mol_opt optimizers inherit from BaseOptimizer and call:
    self.oracle(smiles_list)        -> list[float]
    self.oracle.assign_evaluator()  -> no-op
    self.oracle.finish              -> bool
    self.oracle.mol_buffer          -> {smiles: [score, call_order]}
    len(self.oracle)                -> int
    self.sort_buffer()
    self.mol_buffer
    self.finish
    self.sanitize(mol_list)         -> mol_list (dedup)
    self.log_intermediate()
    self.all_smiles                 -> list[str] (ZINC population)
    self.n_jobs                     -> int
    self.smi_file                   -> None
    self.args.patience              -> int

PMOOracleBridge wraps our PMOOracle to satisfy the Oracle interface.
make_bridged_optimizer() builds a patched mol_opt optimizer instance.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
from rdkit import Chem

_MOL_OPT_ROOT = Path('/home/dominic/mol_opt')


# ── ZINC smiles ───────────────────────────────────────────────────────────────

def load_zinc_smiles(n: Optional[int] = None) -> list[str]:
    """Return ZINC SMILES strings for seeding initial populations.

    Reads from mol_opt's local zinc.tab (fast); falls back to TDC download.
    """
    zinc_tab = _MOL_OPT_ROOT / 'data' / 'zinc.tab'
    if zinc_tab.exists():
        smiles = []
        with open(zinc_tab) as fh:
            for i, line in enumerate(fh):
                if i == 0:
                    continue  # header: "smiles"
                smi = line.strip().strip('"')
                if smi:
                    smiles.append(smi)
                    if n and len(smiles) >= n:
                        break
        return smiles

    # Fallback: TDC download (slower, requires internet on first call)
    from tdc.generation import MolGen
    data = MolGen(name='ZINC')
    smiles = data.get_data()['smiles'].tolist()
    return smiles[:n] if n else smiles


# ── Oracle bridge ─────────────────────────────────────────────────────────────

class _DotDict(dict):
    """Dict with attribute access (mimics argparse Namespace)."""
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError(name)


class PMOOracleBridge:
    """
    Wraps our PMOOracle to look like mol_opt's Oracle class.

    mol_opt algorithms call oracle(smiles_list), check oracle.finish,
    call oracle.assign_evaluator(), etc.  All of these are satisfied here
    by delegating to our PMOOracle.

    Parameters
    ----------
    pmo_oracle : PMOOracle
        Our oracle instance (already configured with budget + checkpointing).
    oracle_key : str
        The property key to use as the scalar score (e.g. 'qed').
    budget : int
        Maximum oracle evaluations.
    freq_log : int
        How often to print progress (every N evaluations).
    """

    def __init__(self, pmo_oracle, oracle_key: str, budget: int,
                 freq_log: int = 100):
        self.pmo_oracle = pmo_oracle
        self.oracle_key = oracle_key
        self.max_oracle_calls = budget
        self.freq_log = freq_log
        # mol_opt format: {canonical_smiles: [score, call_order]}
        self.mol_buffer: dict = {}
        self.last_log = 0
        self.task_label: Optional[str] = None
        # args stub expected by some optimizers
        self.args = _DotDict({
            'output_dir': '.',
            'max_oracle_calls': budget,
            'freq_log': freq_log,
            'patience': 5,
        })
        # Stubs for mol_opt logging helpers (not needed for PMO metrics)
        self.sa_scorer = lambda smis: [0.0] * (len(smis) if isinstance(smis, list) else 1)
        self.diversity_evaluator = lambda smis: 0.0

    # ── mol_opt Oracle interface ───────────────────────────────────────────────

    def assign_evaluator(self, evaluator):
        """No-op: we use PMOOracle directly instead of a TDC evaluator."""
        pass

    def score_smi(self, smi: str) -> float:
        """Evaluate one SMILES, caching in mol_buffer. Returns 0 past budget."""
        if len(self.mol_buffer) >= self.max_oracle_calls:
            return 0.0
        if not smi:
            return 0.0
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return 0.0
        smi = Chem.MolToSmiles(mol)
        if smi not in self.mol_buffer:
            props = self.pmo_oracle.evaluate(smi)
            score = float(props.get(self.oracle_key, 0.0) or 0.0)
            self.mol_buffer[smi] = [score, len(self.mol_buffer) + 1]
        return self.mol_buffer[smi][0]

    def __call__(self, smiles_input):
        """Score one SMILES or a list of SMILES strings."""
        if isinstance(smiles_input, str):
            score = self.score_smi(smiles_input)
            if len(self.mol_buffer) % self.freq_log == 0 and len(self.mol_buffer) > self.last_log:
                self.sort_buffer()
                self._print_progress()
                self.last_log = len(self.mol_buffer)
            return score

        score_list = []
        for smi in smiles_input:
            score_list.append(self.score_smi(smi))
            if (len(self.mol_buffer) % self.freq_log == 0
                    and len(self.mol_buffer) > self.last_log):
                self.sort_buffer()
                self._print_progress()
                self.last_log = len(self.mol_buffer)
        return score_list

    def __len__(self) -> int:
        return len(self.mol_buffer)

    @property
    def finish(self) -> bool:
        return len(self.mol_buffer) >= self.max_oracle_calls

    @property
    def budget(self) -> int:
        return self.max_oracle_calls

    def sort_buffer(self):
        self.mol_buffer = dict(sorted(
            self.mol_buffer.items(), key=lambda kv: kv[1][0], reverse=True
        ))

    def log_intermediate(self, mols=None, scores=None, finish=False):
        """Minimal progress logging (mol_opt algorithms call this internally)."""
        n = len(self.mol_buffer)
        if n == 0:
            return
        sorted_buf = sorted(self.mol_buffer.items(), key=lambda kv: kv[1][0], reverse=True)
        top1 = sorted_buf[0][1][0]
        top10 = float(np.mean([x[1][0] for x in sorted_buf[:10]]))
        print(f'  [{n}/{self.max_oracle_calls}] top1={top1:.4f}  top10={top10:.4f}')

    def save_result(self, suffix=None):
        """No-op: PMOOracle handles its own checkpointing."""
        pass

    def _print_progress(self):
        self.log_intermediate()


# ── Optimizer factory ─────────────────────────────────────────────────────────

def make_bridged_optimizer(optimizer_cls, bridge: PMOOracleBridge,
                           all_smiles: list, args):
    """
    Instantiate a mol_opt optimizer without calling BaseOptimizer.__init__.

    BaseOptimizer.__init__ requires TDC, downloads ZINC, etc.  We bypass it
    via __new__ and manually set the attributes _optimize() actually uses.

    Parameters
    ----------
    optimizer_cls : type
        A mol_opt optimizer class (e.g. GB_GA_Optimizer).
    bridge : PMOOracleBridge
        Our oracle bridge.
    all_smiles : list[str]
        ZINC SMILES for initial population seeding.
    args : namespace/dict-like
        Must have at least 'patience' (int). Other attrs are optional.
    """
    opt = optimizer_cls.__new__(optimizer_cls)

    # Core oracle interface
    opt.oracle = bridge

    # Population source
    opt.all_smiles = all_smiles
    opt.smi_file = None

    # Parallelism — use 1 to keep evaluation ordering deterministic
    opt.n_jobs = 1

    # args namespace (patience controls early-stopping)
    patience = getattr(args, 'patience', None) or getattr(args, 'ga_patience', 5)
    opt.args = _DotDict({
        'patience': patience,
        'output_dir': getattr(args, 'output_dir', '.'),
        'max_oracle_calls': bridge.max_oracle_calls,
        'freq_log': bridge.freq_log,
        'n_jobs': 1,
    })

    # Stubs for logging helpers (mol_opt uses these in log_intermediate / log_result)
    opt.sa_scorer = bridge.sa_scorer
    opt.diversity_evaluator = bridge.diversity_evaluator
    opt.filter = lambda smis: smis  # pass-through, no PAINS filtering

    # Methods that delegate to oracle
    opt.sort_buffer = lambda: bridge.sort_buffer()
    opt.log_intermediate = lambda mols=None, scores=None, finish=False: \
        bridge.log_intermediate(mols=mols, scores=scores, finish=finish)

    # Properties as lambdas (can't set @property on instance, so use __class__)
    # Instead, expose as plain attributes that are refreshed each access via
    # a thin wrapper class. Simpler: just override with regular attribute +
    # make them match the oracle state dynamically via __getattribute__.
    # Easiest: monkey-patch on a per-instance subclass.
    opt.__class__ = _make_patched_class(optimizer_cls)

    # Sanitize
    opt._bridge = bridge

    return opt


def _make_patched_class(base_cls):
    """Return a subclass of base_cls with finish and mol_buffer as properties."""
    name = f'_Bridged_{base_cls.__name__}'
    if name in _PATCHED_CLASS_CACHE:
        return _PATCHED_CLASS_CACHE[name]

    class _BridgedOptimizer(base_cls):
        """Dynamically generated subclass that wires finish/mol_buffer to the bridge."""

        @property
        def finish(self):
            return self._bridge.finish

        @property
        def mol_buffer(self):
            return self._bridge.mol_buffer

        def sanitize(self, mol_list):
            new_mol_list = []
            smiles_set = set()
            for mol in mol_list:
                if mol is not None:
                    try:
                        smiles = Chem.MolToSmiles(mol)
                        if smiles and smiles not in smiles_set:
                            smiles_set.add(smiles)
                            new_mol_list.append(mol)
                    except ValueError:
                        pass
            return new_mol_list

    _BridgedOptimizer.__name__ = name
    _BridgedOptimizer.__qualname__ = name
    _PATCHED_CLASS_CACHE[name] = _BridgedOptimizer
    return _BridgedOptimizer


_PATCHED_CLASS_CACHE: dict = {}
