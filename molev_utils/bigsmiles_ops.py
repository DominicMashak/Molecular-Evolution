"""
BigSMILES polymer genotype support.

Encodes polymers using a canonical BigSMILES string:

    C{[>][<]<RU_SMILES>[>];[<][]}|uniform(lo, hi)|

where <RU_SMILES> is the standalone SMILES of the repeat unit (e.g. CC(C) for
propylene), and the |uniform(lo, hi)| suffix controls the target molecular weight
(g/mol) of the sampled oligomer.

For copolymers, multiple repeat units are comma-separated:

    C{[>][<]<RU1>[>],[<]<RU2>[>];[<][]}|uniform(lo, hi)|

Mutations apply the same 7 SMILES-level operations (MoleculeMutator types 1-7)
to the repeat unit fragment.  Sampling uses gbigsmiles to draw a molecule from
the ensemble, returning the SMILES with the highest QED from n_samples draws.
"""

import re
import random
from typing import List, Optional, Tuple

import numpy as np
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import Descriptors, QED

from molev_utils.molecule_ops import MoleculeMutator

RDLogger.DisableLog('rdApp.*')

# Regex: extract the stochastic block (between { and }) and optional distribution
_BLOCK_RE = re.compile(r'C\{([^}]+)\}(?:\|([^|]+)\|)?')
# Extract each RU SMILES from the [>] section before the semicolon
_RU_RE = re.compile(r'\[<\](.*?)\[>\]')


class BigSMILESMutator:
    """
    Polymer-level mutation and molecule sampling for BigSMILES genotypes.

    Genotypes are canonical BigSMILES strings of the form:
        C{[>][<]<RU>[>];[<][]}|uniform(lo, hi)|

    Mutation applies one of the standard 7 SMILES mutation types to a random
    repeat unit fragment, using the project's existing MoleculeMutator.

    Sampling draws n_samples molecules from the BigSMILES ensemble (via gbigsmiles)
    and returns the SMILES with the highest QED.  Chain length is controlled via
    a target degree of polymerisation (dp), converted to a molecular weight range
    using the repeat unit MW.

    Args:
        atom_set: Atom set for validation and mutation ('nlo', 'drug', ...).
                  Forwarded to MoleculeMutator — any atom set key works.
        n_samples: Number of oligomers sampled per decode call (best QED kept).
        dp: Target degree of polymerisation for sampling.  If None, use
            the distribution already embedded in the BigSMILES string
            (or fall back to dp=3 if none is present).
    """

    def __init__(self, atom_set: str = 'nlo', n_samples: int = 3,
                 dp: Optional[int] = None):
        self.atom_set = atom_set
        self.n_samples = n_samples
        self.dp = dp
        self._smiles_mutator = MoleculeMutator(atom_set=atom_set)

    # ------------------------------------------------------------------
    # Canonical string helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse(s: str) -> Optional[dict]:
        """Parse a canonical BigSMILES string.

        Returns dict with keys 'repeat_units' (list[str]) and
        'distribution' (str | None), or None if the string is not canonical.
        """
        m = _BLOCK_RE.search(s)
        if not m:
            return None
        block = m.group(1)          # everything inside { }
        distribution = m.group(2)   # inside | | (or None)
        # Extract repeat units (everything before the first ';')
        ru_section = block.split(';')[0]
        repeat_units = _RU_RE.findall(ru_section)
        if not repeat_units:
            return None
        return {'repeat_units': repeat_units, 'distribution': distribution}

    @staticmethod
    def _build(repeat_units: List[str], distribution: Optional[str] = None) -> str:
        """Reconstruct a canonical BigSMILES string from components."""
        ru_str = ','.join(f'[<]{ru}[>]' for ru in repeat_units)
        s = 'C{[>]' + ru_str + ';[<][]}'
        if distribution:
            s += f'|{distribution}|'
        return s

    def _compute_distribution(self, ru_smiles: str,
                               dp: Optional[int]) -> Optional[str]:
        """
        Return a uniform distribution string targeting the given DP.

        Converts DP to a molecular weight range using the repeat unit MW
        (including implicit hydrogens) plus a C endcap (~16 g/mol).
        Uses ±30 % around the target to allow natural length variation.
        """
        effective_dp = dp if dp is not None else (self.dp if self.dp is not None else 3)
        mol = Chem.MolFromSmiles(ru_smiles)
        if mol is None:
            mw_target = effective_dp * 50.0
        else:
            mw_target = effective_dp * Descriptors.MolWt(mol) + 16.0
        lo = mw_target * 0.70
        hi = mw_target * 1.30
        return f'uniform({lo:.1f},{hi:.1f})'

    # ------------------------------------------------------------------
    # Fragment validation (relaxed — repeat units are small fragments)
    # ------------------------------------------------------------------

    def _validate_fragment(self, smiles: str,
                           min_atoms: int = 2,
                           max_atoms: int = 15) -> bool:
        """Validate a SMILES repeat unit fragment.

        Uses a relaxed atom count (≥ 2) because small repeat units like
        CC (ethylene) are chemically valid polymers.  Enforces atom set
        membership and RDKit sanitizability.
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False
            if not (min_atoms <= mol.GetNumAtoms() <= max_atoms):
                return False
            allowed = self._smiles_mutator.allowed_atomic_numbers
            for atom in mol.GetAtoms():
                if atom.GetAtomicNum() not in allowed:
                    return False
                if atom.GetFormalCharge() != 0:
                    return False
            # No triple bonds
            for bond in mol.GetBonds():
                if bond.GetBondType() == Chem.BondType.TRIPLE:
                    return False
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def mutate(self, bigsmiles_str: str, mutation_type: int) -> Optional[str]:
        """Apply one SMILES mutation (type 1-7) to a random repeat unit.

        The mutation operates on the repeat unit SMILES fragment using the
        same MoleculeMutator operations as for standard SMILES genotypes.
        Returns a new canonical BigSMILES string, or None if the mutation
        produces an invalid fragment.
        """
        parsed = self._parse(bigsmiles_str)
        if parsed is None:
            return None
        repeat_units = list(parsed['repeat_units'])  # copy
        if not repeat_units:
            return None

        # Pick a random repeat unit to mutate
        idx = random.randrange(len(repeat_units))
        ru = repeat_units[idx]

        mutated = self._smiles_mutator.mutate(ru, mutation_type)
        if mutated is None:
            return None
        if not self._validate_fragment(mutated):
            return None

        repeat_units[idx] = mutated
        # Recompute distribution for the new repeat unit
        new_dist = self._compute_distribution(repeat_units[0], self.dp)
        return self._build(repeat_units, new_dist)

    def to_smiles(self, bigsmiles_str: str,
                  n_samples: Optional[int] = None,
                  dp: Optional[int] = None) -> Optional[str]:
        """Sample molecules from a BigSMILES ensemble, return best-QED SMILES.

        Args:
            bigsmiles_str: Canonical BigSMILES string.
            n_samples: How many oligomers to draw (default: self.n_samples).
            dp: Target degree of polymerisation; overrides self.dp.

        Returns:
            SMILES string of the sample with the highest QED, or None if
            no valid sample could be generated.
        """
        try:
            import gbigsmiles
        except ImportError:
            raise ImportError(
                "gbigsmiles is required for BigSMILES encoding. "
                "Install with: pip install gbigsmiles"
            )

        effective_n = n_samples if n_samples is not None else self.n_samples
        effective_dp = dp if dp is not None else self.dp

        parsed = self._parse(bigsmiles_str)
        if parsed is None:
            return None
        repeat_units = parsed['repeat_units']
        if not repeat_units:
            return None

        # Build sampling string with MW distribution
        dist = self._compute_distribution(repeat_units[0], effective_dp)
        bs_str = self._build(repeat_units, dist)

        try:
            bs = gbigsmiles.BigSmiles.make(bs_str)
            gen = bs.get_generating_graph()
            ag = gen.get_atom_graph()
        except Exception:
            return None

        candidates: List[Tuple[float, str]] = []
        for seed in range(effective_n * 4):  # oversample to fill n_samples
            try:
                rng = np.random.default_rng(seed)
                mg = ag.sample_mol_graph(rng=rng)
                mol = gbigsmiles.mol_graph_to_rdkit_mol(mg)
                if mol is None:
                    continue
                if Chem.SanitizeMol(mol, catchErrors=True) != 0:
                    continue
                smi = Chem.MolToSmiles(mol)
                if smi is None:
                    continue
                # Validate with existing MoleculeMutator (full molecule checks)
                if not self._smiles_mutator.validate(smi, max_atoms=60):
                    continue
                qed_score = QED.qed(mol)
                candidates.append((qed_score, smi))
                if len(candidates) >= effective_n:
                    break
            except Exception:
                continue

        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    def validate(self, bigsmiles_str: str, max_atoms: int = 60) -> bool:
        """Validate a canonical BigSMILES string.

        Checks canonical format and samples a DP=2 oligomer to verify
        that the repeat unit produces valid molecules.
        """
        parsed = self._parse(bigsmiles_str)
        if parsed is None or not parsed['repeat_units']:
            return False
        # Quick fragment check on each RU
        for ru in parsed['repeat_units']:
            if not self._validate_fragment(ru):
                return False
        # Verify sampling works (use dp=2 for speed)
        smi = self.to_smiles(bigsmiles_str, n_samples=1, dp=2)
        return smi is not None


# ------------------------------------------------------------------
# Module-level helpers (mirror selfies_ops.py interface)
# ------------------------------------------------------------------

def smiles_to_bigsmiles(smiles: str) -> Optional[str]:
    """Wrap a SMILES repeat unit in canonical BigSMILES form.

    The resulting string has no embedded distribution; a distribution
    is added automatically by BigSMILESMutator.to_smiles() at sample time.
    """
    if smiles is None:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
    except Exception:
        return None
    return f'C{{[>][<]{smiles}[>];[<][]}}'


def bigsmiles_to_smiles(bigsmiles_str: str,
                        n_samples: int = 3,
                        dp: Optional[int] = None,
                        atom_set: str = 'nlo') -> Optional[str]:
    """Sample a SMILES from a BigSMILES ensemble.

    Convenience wrapper around BigSMILESMutator.to_smiles().
    """
    m = BigSMILESMutator(atom_set=atom_set, n_samples=n_samples, dp=dp)
    return m.to_smiles(bigsmiles_str)
