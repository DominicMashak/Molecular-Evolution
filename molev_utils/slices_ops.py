"""
SLICES crystal genotype support.

Encodes periodic crystal structures using the SLICES (Simplified Line-Input
Crystal-Encoding System) string format, strategy 4:

    elem1 elem2 ... elemN  i1 j1 pbc1  i2 j2 pbc2  ...

where pbc tokens are 3-char strings of {o, +, -} encoding periodic boundary
conditions (+1, 0, -1) in x, y, z directions.

Mutations operate at the token level (no RDKit).  Property evaluation uses
matgl (PyTorch M3GNet) for formation energy prediction, bypassing the
TensorFlow-based m3gnet that ships with the slices pip package.

The SLICES library (https://github.com/xiaohang007/SLICES) imports tensorflow
at module level.  We mock TF in sys.modules before the first import of
slices.core so the reconstruction code (which uses scipy L-BFGS-B) works
without a TF installation.
"""

import sys
import random
from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# TensorFlow mock — must be applied before importing slices.core
# ---------------------------------------------------------------------------

def _mock_tf_for_slices() -> None:
    """Pre-populate sys.modules with lightweight mocks for tensorflow / m3gnet.

    slices/core.py does:
        import tensorflow as tf
        import m3gnet.models
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)

    at module level.  The actual structure reconstruction (to_structures) uses
    scipy L-BFGS-B, not TF.  The relax() call wraps M3GNet IAPs — when it
    fails (mocked), to_structures() falls back to the ZL*-optimised structure.
    """
    from unittest.mock import MagicMock

    if 'tensorflow' not in sys.modules:
        tf_mock = MagicMock()
        tf_mock.config.threading.set_inter_op_parallelism_threads = \
            MagicMock(return_value=None)
        tf_mock.config.threading.set_intra_op_parallelism_threads = \
            MagicMock(return_value=None)
        sys.modules['tensorflow'] = tf_mock

    for mod_name in ('m3gnet', 'm3gnet.models'):
        if mod_name not in sys.modules:
            sys.modules[mod_name] = MagicMock()


# ---------------------------------------------------------------------------
# Element sets (analogous to molecule atom_set)
# ---------------------------------------------------------------------------

ELEMENT_SETS: dict = {
    'oxides': {
        'Li', 'Na', 'K', 'Ca', 'Mg', 'Al', 'Si', 'Ti', 'V', 'Cr', 'Mn',
        'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Sn',
        'Ba', 'La', 'Ce', 'W', 'Pb', 'Bi', 'O',
    },
    'semiconductor': {
        'Si', 'Ge', 'C', 'Sn', 'Ga', 'As', 'In', 'P', 'Al', 'Sb',
        'Cd', 'S', 'Se', 'Te', 'Zn', 'Hg', 'N',
    },
    'halide_perovskite': {
        'Cs', 'Rb', 'K', 'Na', 'Li', 'Pb', 'Sn', 'Ge', 'I', 'Br', 'Cl',
    },
    # Advanced semiconductor element palette.
    # Covers all major families used in modern semiconductor research:
    #   Group IV elemental/alloy:  C (diamond), Si, Ge, Sn, SiC, SiGe
    #   III-V nitrides (wide-gap): B, Al, Ga, In + N  (AlN, GaN, InN, BN)
    #   III-V arsenides/phosphides: Ga, In, Al + As, P, Sb
    #   II-VI chalcogenides:       Zn, Cd, Hg + S, Se, Te  (ZnS, CdTe, HgTe)
    #   Wide-gap oxides:           Ga (β-Ga₂O₃), In (In₂O₃/ITO), Sn (SnO₂),
    #                              Zn (ZnO), O
    #   Transition-metal dichalcogenides (TMD): Mo, W + S, Se, Te
    #   Chalcopyrite/kesterite:    Cu, Ag, In, Ga + S, Se  (CuInSe₂, CZTS)
    #   Narrow-gap / topological:  Bi, Sb, Te  (Bi₂Te₃, Sb₂Te₃, BiSb)
    'semiconductors': {
        # Group IV
        'C', 'Si', 'Ge', 'Sn',
        # Group III
        'B', 'Al', 'Ga', 'In',
        # Group V
        'N', 'P', 'As', 'Sb', 'Bi',
        # Group II
        'Zn', 'Cd', 'Hg',
        # Group VI / chalcogens
        'O', 'S', 'Se', 'Te',
        # Transition metals (TMDs and wide-gap oxides)
        'Mo', 'W',
        # Chalcopyrite / kesterite cations
        'Cu', 'Ag',
        # Rare-earth / nitride alloy dopants
        'Sc',
    },
}

_PBC_CHARS: List[str] = ['o', '+', '-']

# ---------------------------------------------------------------------------
# Valence-group constraints for semiconductor element substitution.
# Only swap within a group to preserve charge balance and coordination:
#   Group IV   (diamond cubic / alloys): C, Si, Ge, Sn
#   Group III  (cation in III-V):        B, Al, Ga, In
#   Group V    (anion in III-V):         N, P, As, Sb, Bi
#   Group II   (cation in II-VI):        Zn, Cd, Hg
#   Group VI   (anion in II-VI):         S, Se, Te
#   Transition (TMDs):                   Mo, W
#   Chalcogen  (shared with VI):         O (oxide group)
#   Halides    (perovskites):            I, Br, Cl, F
#   A-cations  (perovskites):            Cs, Rb, K, Na, Li
#   B-cations  (perovskites):            Pb, Sn, Ge
# Elements not listed fall back to unconstrained substitution.
# ---------------------------------------------------------------------------

_VALENCE_GROUPS: dict = {
    # Group IV
    'C': 'IV', 'Si': 'IV', 'Ge': 'IV', 'Sn': 'IV',
    # Group III (III-V cation)
    'B': 'III', 'Al': 'III', 'Ga': 'III', 'In': 'III',
    # Group V (III-V anion)
    'N': 'V', 'P': 'V', 'As': 'V', 'Sb': 'V', 'Bi': 'V',
    # Group II (II-VI cation)
    'Zn': 'II', 'Cd': 'II', 'Hg': 'II',
    # Group VI chalcogens (II-VI anion)
    'S': 'VI', 'Se': 'VI', 'Te': 'VI',
    # Transition metals (TMDs)
    'Mo': 'TM', 'W': 'TM',
    # Oxygen (oxide)
    'O': 'O',
    # Halides
    'F': 'HAL', 'Cl': 'HAL', 'Br': 'HAL', 'I': 'HAL',
    # Perovskite A-cations
    'Cs': 'A_PERO', 'Rb': 'A_PERO', 'K': 'A_PERO', 'Na': 'A_PERO', 'Li': 'A_PERO',
    # Perovskite B-cations
    'Pb': 'B_PERO',
}

# Inverse map: valence group → list of elements in that group
_GROUP_TO_ELEMENTS: dict = {}
for _el, _grp in _VALENCE_GROUPS.items():
    _GROUP_TO_ELEMENTS.setdefault(_grp, []).append(_el)


def _valence_group_candidates(element: str, allowed_elements: List[str]) -> List[str]:
    """Return allowed replacement elements in the same valence group as *element*.

    Falls back to all allowed elements (excluding the element itself) if the
    element has no group mapping or if the group has only one allowed member.
    """
    grp = _VALENCE_GROUPS.get(element)
    if grp is not None:
        same_group = [e for e in _GROUP_TO_ELEMENTS.get(grp, [])
                      if e in allowed_elements and e != element]
        if same_group:
            return same_group
    # Fallback: unconstrained (element not in map, or only member of its group)
    return [e for e in allowed_elements if e != element]


# ---------------------------------------------------------------------------
# SLICESMutator
# ---------------------------------------------------------------------------

class SLICESMutator:
    """Crystal-level mutation for SLICES strategy-4 genotypes.

    Genotype format (strategy 4, no space-group prefix)::

        Ti O O  0 1 ooo  0 2 +oo  1 2 o+o  0 1 oo+

    Six mutation types:
        1 — substitute_element : replace one atom symbol
        2 — add_site           : append a new atom + one connecting edge
        3 — remove_site        : delete an atom and all its edges
        4 — change_pbc         : flip one character in a random edge's pbc string
        5 — add_edge           : insert a new edge between two existing atoms
        6 — remove_edge        : delete a random edge

    Args:
        element_set: Key into ELEMENT_SETS ('oxides', 'semiconductor',
                     'semiconductors', 'halide_perovskite').
                     'semiconductors' is the broad advanced-semiconductor palette
                     (III-V, II-VI, wide-gap oxides, TMDs, chalcopyrites).
        min_sites:   Minimum number of atomic sites.
        max_sites:   Maximum number of atomic sites.
    """

    def __init__(self, element_set: str = 'oxides',
                 min_sites: int = 2, max_sites: int = 20) -> None:
        self.element_set = element_set
        self.min_sites = min_sites
        self.max_sites = max_sites
        self.allowed_elements: List[str] = sorted(
            ELEMENT_SETS.get(element_set, ELEMENT_SETS['oxides'])
        )

    # ------------------------------------------------------------------
    # Parsing / building
    # ------------------------------------------------------------------

    @staticmethod
    def _parse(slices_str: str) -> Optional[dict]:
        """Parse a SLICES strategy-4 string.

        Returns ``{'atoms': [str], 'edges': [(int, int, str)]}`` or None.

        Handles strings with or without a space-group prefix (the prefix
        tokens are skipped — they are non-element, non-integer tokens before
        the first valid element symbol).
        """
        try:
            from pymatgen.core import Element as PMGElement
        except ImportError:
            return None

        tokens = slices_str.strip().split()
        if not tokens:
            return None

        # Find first valid element token (possibly after a space-group prefix)
        first_elem_idx: Optional[int] = None
        for i, tok in enumerate(tokens):
            try:
                PMGElement(tok)
                first_elem_idx = i
                break
            except Exception:
                continue

        if first_elem_idx is None:
            return None

        # Atoms end at the first integer token after the element section
        num_atoms: Optional[int] = None
        for i in range(first_elem_idx, len(tokens)):
            if tokens[i].isnumeric():
                num_atoms = i - first_elem_idx
                break

        if num_atoms is None:
            # No edge section — valid (atom-only, will fail graph checks later)
            return {'atoms': tokens[first_elem_idx:], 'edges': []}

        if num_atoms == 0:
            return None

        atoms = tokens[first_elem_idx: first_elem_idx + num_atoms]
        edge_start = first_elem_idx + num_atoms
        remaining = tokens[edge_start:]

        if len(remaining) % 3 != 0:
            return None

        edges: List[Tuple[int, int, str]] = []
        for i in range(0, len(remaining), 3):
            try:
                a = int(remaining[i])
                b = int(remaining[i + 1])
                pbc = remaining[i + 2]
            except (ValueError, IndexError):
                return None
            if len(pbc) != 3 or not all(c in 'o+-' for c in pbc):
                return None
            if a < 0 or a >= num_atoms or b < 0 or b >= num_atoms:
                return None
            edges.append((a, b, pbc))

        return {'atoms': atoms, 'edges': edges}

    @staticmethod
    def _build(atoms: List[str], edges: List[Tuple[int, int, str]]) -> str:
        """Reconstruct a SLICES string (no space-group prefix)."""
        parts: List[str] = list(atoms)
        for a, b, pbc in edges:
            parts.extend([str(a), str(b), pbc])
        return ' '.join(parts)

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def mutate(self, slices_str: str, mutation_type: int) -> Optional[str]:
        """Apply one of 6 mutations to a SLICES string.

        Returns the mutated string, or None if the mutation is not applicable
        (e.g. removing a site from a 2-atom structure).
        """
        parsed = self._parse(slices_str)
        if parsed is None:
            return None

        atoms = list(parsed['atoms'])
        edges = list(parsed['edges'])
        n = len(atoms)

        if mutation_type == 1:
            return self._substitute_element(atoms, edges)
        if mutation_type == 2:
            return self._add_site(atoms, edges, n)
        if mutation_type == 3:
            return self._remove_site(atoms, edges, n)
        if mutation_type == 4:
            return self._change_pbc(atoms, edges)
        if mutation_type == 5:
            return self._add_edge(atoms, edges, n)
        if mutation_type == 6:
            return self._remove_edge(atoms, edges)
        return None

    def _substitute_element(self, atoms: List[str],
                             edges: List[Tuple]) -> Optional[str]:
        if not atoms:
            return None
        idx = random.randrange(len(atoms))
        candidates = _valence_group_candidates(atoms[idx], self.allowed_elements)
        if not candidates:
            return None
        atoms[idx] = random.choice(candidates)
        return self._build(atoms, edges)

    def _add_site(self, atoms: List[str], edges: List[Tuple],
                  n: int) -> Optional[str]:
        if n >= self.max_sites:
            return None
        # Pick a new element compatible with the most common valence group
        # already present, so added sites maintain semiconductor stoichiometry.
        if atoms:
            ref_elem = random.choice(atoms)
            candidates = _valence_group_candidates(ref_elem, self.allowed_elements)
            # Include ref_elem itself so same element can be added
            candidates = candidates or self.allowed_elements
            new_elem = random.choice(candidates + [ref_elem])
        else:
            new_elem = random.choice(self.allowed_elements)
        new_idx = n
        atoms.append(new_elem)
        existing = random.randrange(n)
        pbc = ''.join(random.choices(_PBC_CHARS, k=3))
        edges.append((existing, new_idx, pbc))
        return self._build(atoms, edges)

    def _remove_site(self, atoms: List[str], edges: List[Tuple],
                     n: int) -> Optional[str]:
        if n <= self.min_sites:
            return None
        idx = random.randrange(n)
        atoms.pop(idx)
        # Remove all edges involving this atom; renumber remaining indices
        edges = [
            (a - (1 if a > idx else 0),
             b - (1 if b > idx else 0),
             pbc)
            for a, b, pbc in edges
            if a != idx and b != idx
        ]
        return self._build(atoms, edges)

    def _change_pbc(self, atoms: List[str],
                    edges: List[Tuple]) -> Optional[str]:
        if not edges:
            return None
        idx = random.randrange(len(edges))
        a, b, pbc = edges[idx]
        pos = random.randrange(3)
        new_char = random.choice([c for c in _PBC_CHARS if c != pbc[pos]])
        edges[idx] = (a, b, pbc[:pos] + new_char + pbc[pos + 1:])
        return self._build(atoms, edges)

    def _add_edge(self, atoms: List[str], edges: List[Tuple],
                  n: int) -> Optional[str]:
        if n < 2:
            return None
        a = random.randrange(n)
        b = random.randrange(n)
        pbc = ''.join(random.choices(_PBC_CHARS, k=3))
        edges.append((a, b, pbc))
        return self._build(atoms, edges)

    def _remove_edge(self, atoms: List[str],
                     edges: List[Tuple]) -> Optional[str]:
        if len(edges) <= 1:
            return None
        idx = random.randrange(len(edges))
        edges.pop(idx)
        return self._build(atoms, edges)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, slices_str: str) -> bool:
        """Validate a SLICES string.

        Runs basic format checks, then uses the SLICES library's
        ``check_SLICES()`` for full graph-topological validation (requires
        rank H1(X,Z) ≥ 3 for 3-D embedding).  Falls back to basic checks
        if the library is unavailable.
        """
        parsed = self._parse(slices_str)
        if parsed is None:
            return False

        atoms = parsed['atoms']
        edges = parsed['edges']
        n = len(atoms)

        if not (self.min_sites <= n <= self.max_sites):
            return False
        if not edges:
            return False
        for elem in atoms:
            if elem not in self.allowed_elements:
                return False

        try:
            _mock_tf_for_slices()
            from slices.core import SLICES as _SLIBBackend
            backend = _SLIBBackend(relax_model=None, graph_method='econnn')
            return backend.check_SLICES(slices_str, strategy=4)
        except Exception:
            return True  # fall back to basic checks

    # ------------------------------------------------------------------
    # Structure decoding
    # ------------------------------------------------------------------

    def to_structure(self, slices_str: str):
        """Convert a SLICES string to a pymatgen Structure.

        Uses the SLICES library's SLICES2structure() (strategy 4).  TF is
        mocked; structure reconstruction uses scipy L-BFGS-B and gracefully
        falls back to the ZL*-optimised structure if M3GNet relaxation fails.

        Returns:
            pymatgen.core.Structure or None on failure.
        """
        try:
            _mock_tf_for_slices()
            from slices.core import SLICES as _SLIBBackend
            backend = _SLIBBackend(relax_model=None, graph_method='econnn')
            structure, _energy = backend.SLICES2structure(
                slices_str, strategy=4)
            return structure
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Seed generation
    # ------------------------------------------------------------------

    def generate_seeds(self, element_set: str = None,
                       n: int = 30) -> List[str]:
        """Generate diverse seed SLICES strings for initial population.

        Builds a set of prototype crystal structures using pymatgen,
        converts each to a SLICES string, then diversifies via random
        mutations to reach the target count.

        Args:
            element_set: Override the instance element set.
            n:           Target number of seed strings.

        Returns:
            List of validated SLICES strings (may be fewer than *n* if
            prototypes fail to convert).
        """
        eset = element_set or self.element_set

        protos = _build_prototype_structures(eset)
        base_seeds: List[str] = []
        for struct in protos:
            s = structure_to_slices(struct)
            if s and self.validate(s):
                base_seeds.append(s)

        result = list(base_seeds)
        attempts = 0
        max_attempts = n * 30

        while len(result) < n and attempts < max_attempts and base_seeds:
            attempts += 1
            base = random.choice(base_seeds)
            mutation_type = random.randint(1, 6)
            mutated = self.mutate(base, mutation_type)
            if mutated and mutated not in result and self.validate(mutated):
                result.append(mutated)

        return result


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def structure_to_slices(structure) -> Optional[str]:
    """Convert a pymatgen Structure to a SLICES string (no space-group prefix).

    Uses the SLICES library's structure2SLICES() method (strategy 4).
    The space-group prefix is stripped so that the resulting string is in
    the clean mutation-friendly format.

    Returns:
        SLICES string or None on failure.
    """
    try:
        _mock_tf_for_slices()
        from slices.core import SLICES as _SLIBBackend
        backend = _SLIBBackend(relax_model=None, graph_method='econnn')
        raw = backend.structure2SLICES(structure, strategy=4)
        # Re-parse and rebuild to strip any space-group prefix
        parsed = SLICESMutator._parse(raw)
        if parsed is None:
            return None
        return SLICESMutator._build(parsed['atoms'], parsed['edges'])
    except Exception:
        return None


def crystal_error_props() -> dict:
    """Return a zero-valued crystal props dict for failed evaluations.

    Keys mirror the molecule props dict so existing archive serialisation
    code (which references num_atoms / num_bonds) does not crash.
    """
    return {
        'slices': None,
        'smiles': None,
        'num_atoms': 0,
        'num_bonds': 0,
        'n_sites': 0,
        'n_species': 0,
        'volume': 0.0,
        'density': 0.0,
        'spacegroup': 0,
        'formation_energy': 0.0,
        'bandgap': 0.0,
        'error': 'Invalid',
    }


# ---------------------------------------------------------------------------
# Prototype structure builder
# ---------------------------------------------------------------------------

def _build_prototype_structures(element_set: str) -> list:
    """Build a list of pymatgen prototype Structure objects.

    Returns an empty list if pymatgen is unavailable.
    """
    try:
        from pymatgen.core import Structure, Lattice
    except ImportError:
        return []

    structures = []

    def _try_add(fn):
        try:
            structures.append(fn())
        except Exception:
            pass

    if element_set == 'oxides':
        # Rock-salt (2 atoms, cubic)
        for a_elem, b_elem, a_val in [
            ('Na', 'Cl', 5.64), ('Mg', 'O', 4.21), ('Ca', 'O', 4.80),
            ('Fe', 'O', 4.33), ('Ni', 'O', 4.18), ('Co', 'O', 4.26),
        ]:
            _try_add(lambda ae=a_elem, be=b_elem, av=a_val: Structure(
                Lattice.cubic(av), [ae, be],
                [[0, 0, 0], [0.5, 0.5, 0.5]]))

        # Cubic perovskite ABO3 (5 atoms)
        for a_elem, b_elem, a_val in [
            ('Sr', 'Ti', 3.91), ('Ba', 'Ti', 4.00), ('Ca', 'Ti', 3.84),
            ('Sr', 'Zr', 4.15), ('Ba', 'Zr', 4.23),
        ]:
            _try_add(lambda ae=a_elem, be=b_elem, av=a_val: Structure(
                Lattice.cubic(av),
                [ae, be, 'O', 'O', 'O'],
                [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5],
                 [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]]))

        # Rutile TiO2 (6 atoms, tetragonal)
        _try_add(lambda: Structure(
            Lattice.tetragonal(4.59, 2.96),
            ['Ti', 'Ti', 'O', 'O', 'O', 'O'],
            [[0, 0, 0], [0.5, 0.5, 0.5],
             [0.31, 0.31, 0.0], [0.69, 0.69, 0.0],
             [0.81, 0.19, 0.5], [0.19, 0.81, 0.5]]))

        # ZnO wurtzite (4 atoms, hexagonal)
        _try_add(lambda: Structure(
            Lattice.hexagonal(3.25, 5.21),
            ['Zn', 'Zn', 'O', 'O'],
            [[0.333, 0.667, 0.0], [0.667, 0.333, 0.5],
             [0.333, 0.667, 0.375], [0.667, 0.333, 0.875]]))

        # Corundum Al2O3 (simplified, hexagonal, 5 atoms)
        _try_add(lambda: Structure(
            Lattice.hexagonal(4.76, 12.99),
            ['Al', 'Al', 'O', 'O', 'O'],
            [[0, 0, 0.35], [0, 0, 0.85],
             [0.31, 0, 0.25], [0.0, 0.31, 0.25], [0.69, 0.69, 0.25]]))

        # Fluorite CaF2 — here using BaO2 (similar structure, element set)
        _try_add(lambda: Structure(
            Lattice.cubic(5.54),
            ['Ba', 'O', 'O'],
            [[0, 0, 0], [0.25, 0.25, 0.25], [0.75, 0.75, 0.75]]))

    elif element_set == 'semiconductor':
        # Diamond cubic (8 atoms)
        for elem, a in [('Si', 5.43), ('Ge', 5.66), ('C', 3.57)]:
            _try_add(lambda el=elem, av=a: Structure(
                Lattice.cubic(av),
                [el] * 8,
                [[0.0, 0.0, 0.0], [0.5, 0.5, 0.0],
                 [0.5, 0.0, 0.5], [0.0, 0.5, 0.5],
                 [0.25, 0.25, 0.25], [0.75, 0.75, 0.25],
                 [0.75, 0.25, 0.75], [0.25, 0.75, 0.75]]))

        # Zinc-blende (2-atom primitive, cubic)
        for a_elem, b_elem, a_val in [
            ('Ga', 'As', 5.65), ('In', 'P', 5.87), ('Zn', 'S', 5.41),
            ('Cd', 'Te', 6.48), ('Al', 'P', 5.46),
        ]:
            _try_add(lambda ae=a_elem, be=b_elem, av=a_val: Structure(
                Lattice.cubic(av), [ae, be],
                [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]))

        # Wurtzite GaN (4 atoms, hexagonal)
        _try_add(lambda: Structure(
            Lattice.hexagonal(3.19, 5.19),
            ['Ga', 'Ga', 'N', 'N'],
            [[0.333, 0.667, 0.0], [0.667, 0.333, 0.5],
             [0.333, 0.667, 0.375], [0.667, 0.333, 0.875]]))

    elif element_set == 'semiconductors':
        # --- Group IV elemental (diamond cubic, 8 atoms) ---
        for elem, a in [('Si', 5.43), ('Ge', 5.66), ('C', 3.57), ('Sn', 6.49)]:
            _try_add(lambda el=elem, av=a: Structure(
                Lattice.cubic(av),
                [el] * 8,
                [[0.0, 0.0, 0.0], [0.5, 0.5, 0.0],
                 [0.5, 0.0, 0.5], [0.0, 0.5, 0.5],
                 [0.25, 0.25, 0.25], [0.75, 0.75, 0.25],
                 [0.75, 0.25, 0.75], [0.25, 0.75, 0.75]]))

        # --- III-V zinc-blende (2-atom primitive) ---
        for a_elem, b_elem, a_val in [
            ('Ga', 'As', 5.65), ('Ga', 'P',  5.45), ('Ga', 'Sb', 6.10),
            ('In', 'P',  5.87), ('In', 'As', 6.06), ('In', 'Sb', 6.48),
            ('Al', 'As', 5.66), ('Al', 'P',  5.46), ('Al', 'Sb', 6.14),
            ('B',  'P',  4.54), ('B',  'As', 4.78),
        ]:
            _try_add(lambda ae=a_elem, be=b_elem, av=a_val: Structure(
                Lattice.cubic(av), [ae, be],
                [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]))

        # --- III-V nitrides (wurtzite, hexagonal, 4 atoms) ---
        for a_elem, a_val, c_val in [
            ('Ga', 3.19, 5.19), ('Al', 3.11, 4.98),
            ('In', 3.54, 5.71), ('B',  2.55, 4.23),
        ]:
            _try_add(lambda ae=a_elem, av=a_val, cv=c_val: Structure(
                Lattice.hexagonal(av, cv),
                [ae, ae, 'N', 'N'],
                [[0.333, 0.667, 0.0], [0.667, 0.333, 0.5],
                 [0.333, 0.667, 0.375], [0.667, 0.333, 0.875]]))

        # --- II-VI zinc-blende ---
        for a_elem, b_elem, a_val in [
            ('Zn', 'S',  5.41), ('Zn', 'Se', 5.67), ('Zn', 'Te', 6.10),
            ('Cd', 'S',  5.82), ('Cd', 'Se', 6.05), ('Cd', 'Te', 6.48),
            ('Hg', 'Te', 6.46),
        ]:
            _try_add(lambda ae=a_elem, be=b_elem, av=a_val: Structure(
                Lattice.cubic(av), [ae, be],
                [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]))

        # --- SiC (zinc-blende, 2-atom primitive) ---
        _try_add(lambda: Structure(
            Lattice.cubic(4.36), ['Si', 'C'],
            [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]))

        # --- Wide-gap oxides (wurtzite ZnO, rutile SnO2, corundum In2O3 proxy) ---
        _try_add(lambda: Structure(
            Lattice.hexagonal(3.25, 5.21), ['Zn', 'Zn', 'O', 'O'],
            [[0.333, 0.667, 0.0], [0.667, 0.333, 0.5],
             [0.333, 0.667, 0.375], [0.667, 0.333, 0.875]]))
        _try_add(lambda: Structure(
            Lattice.tetragonal(4.74, 3.19),
            ['Sn', 'Sn', 'O', 'O', 'O', 'O'],
            [[0, 0, 0], [0.5, 0.5, 0.5],
             [0.31, 0.31, 0.0], [0.69, 0.69, 0.0],
             [0.81, 0.19, 0.5], [0.19, 0.81, 0.5]]))
        _try_add(lambda: Structure(
            Lattice.cubic(5.02), ['In', 'In', 'O', 'O', 'O'],
            [[0, 0, 0], [0.5, 0.5, 0.5],
             [0.25, 0.25, 0.25], [0.75, 0.75, 0.25], [0.5, 0.0, 0.5]]))

        # --- TMDs: MoS2, WS2, MoSe2 (hexagonal layered, 3-atom primitive) ---
        for m_elem, x_elem, a_val, c_val in [
            ('Mo', 'S',  3.16, 12.30), ('W',  'S',  3.15, 12.32),
            ('Mo', 'Se', 3.29, 12.90), ('W',  'Se', 3.28, 12.96),
        ]:
            _try_add(lambda me=m_elem, xe=x_elem, av=a_val, cv=c_val: Structure(
                Lattice.hexagonal(av, cv),
                [me, xe, xe],
                [[0.333, 0.667, 0.25], [0.667, 0.333, 0.125],
                 [0.667, 0.333, 0.375]]))

        # --- Chalcopyrite CuInSe2 (tetragonal, 8 atoms) ---
        _try_add(lambda: Structure(
            Lattice.tetragonal(5.78, 11.60),
            ['Cu', 'Cu', 'In', 'In', 'Se', 'Se', 'Se', 'Se'],
            [[0.0, 0.5, 0.25], [0.5, 0.0, 0.75],
             [0.0, 0.0, 0.0],  [0.5, 0.5, 0.5],
             [0.25, 0.25, 0.125], [0.75, 0.75, 0.125],
             [0.25, 0.75, 0.375], [0.75, 0.25, 0.375]]))

        # --- Bi2Te3 / Sb2Te3 (rhombohedral, 5 atoms) ---
        for a_elem, a_val, c_val in [
            ('Bi', 4.38, 30.49), ('Sb', 4.26, 30.35),
        ]:
            _try_add(lambda ae=a_elem, av=a_val, cv=c_val: Structure(
                Lattice.hexagonal(av, cv),
                [ae, ae, 'Te', 'Te', 'Te'],
                [[0, 0, 0.4], [0, 0, 0.6],
                 [0, 0, 0.0], [0, 0, 0.21], [0, 0, 0.79]]))

    elif element_set == 'halide_perovskite':
        # Cubic ABX3 perovskite (5 atoms)
        for a_elem, b_elem, x_elem, a_val in [
            ('Cs', 'Pb', 'I', 6.25), ('Cs', 'Sn', 'I', 6.18),
            ('Rb', 'Pb', 'Br', 5.95), ('K', 'Pb', 'Cl', 5.61),
            ('Cs', 'Ge', 'I', 6.10), ('Cs', 'Pb', 'Br', 5.87),
        ]:
            _try_add(lambda ae=a_elem, be=b_elem, xe=x_elem, av=a_val:
                     Structure(
                         Lattice.cubic(av),
                         [ae, be, xe, xe, xe],
                         [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5],
                          [0.5, 0.5, 0.0], [0.5, 0.0, 0.5],
                          [0.0, 0.5, 0.5]]))

    return structures
