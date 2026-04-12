"""
molev_utils/problem_config.py
─────────────────────────────
Central registry for problem-specific configuration: property bounds,
objective directions, reference points, and archive measure bounds.

Usage
-----
    from molev_utils.problem_config import get_preset, list_presets, PROPERTY_BOUNDS

    # Load a built-in preset
    cfg = get_preset('nlo_4obj')

    # Inspect
    print(cfg.objective_keys)      # ['beta_gamma_ratio', ...]
    print(cfg.reference_point)     # [0.0, 0.0, 500.0, 100.0]
    print(cfg.optimize_directions) # [('max', None), ('min', None), ...]

    # Tinker — originals are never mutated
    cfg2 = cfg.with_objectives('beta_mean', 'homo_lumo_gap')
    cfg3 = cfg.override_reference('alpha_range_distance', 300.0)
    cfg4 = cfg.override_bounds('beta_mean', (0.0, 200.0))

    # Register your own preset
    from molev_utils.problem_config import register_preset, ProblemConfig, ObjectiveSpec, MeasureSpec
    my_cfg = ProblemConfig(
        name='my_nlo_exp',
        description='Custom 2-obj NLO experiment',
        atom_set='nlo',
        fitness_mode='qc',
        objectives=[
            ObjectiveSpec('beta_mean',     'maximize', (0.0, 200.0), 0.0),
            ObjectiveSpec('homo_lumo_gap', 'maximize', (0.0, 30.0),  0.0),
        ],
        measures=[
            MeasureSpec('num_atoms', (5, 35)),
            MeasureSpec('num_bonds', (4, 40)),
        ],
    )
    register_preset('my_nlo_exp', my_cfg)

Integration in algorithm main.py
---------------------------------
After args = parser.parse_args(), add:

    from molev_utils.problem_config import resolve_from_args
    problem = resolve_from_args(args)
    if problem is not None:
        if args.objectives is None:
            args.objectives = problem.objective_keys
        if args.optimize is None:
            args.optimize = problem.optimize_strings
        if getattr(args, 'reference_point', None) is None:
            args.reference_point = problem.reference_point
        if args.measure_bounds is None:
            args.measure_bounds = problem.measure_bounds_flat
"""
from __future__ import annotations

import copy
import dataclasses
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Canonical property bounds
# Single source of truth — import this instead of defining a local
# _PROPERTY_DEFAULT_BOUNDS dict in each algorithm's main.py.
# ---------------------------------------------------------------------------

PROPERTY_BOUNDS: Dict[str, Tuple[float, float]] = {
    # ── Drug-design descriptors ────────────────────────────────────────────
    'qed':                          (0.0,    1.0),   # Quantitative Estimate of Drug-likeness
    'sa':                           (0.0,    1.0),   # inverted SA score (1 = easy to synthesize)
    'sa_score':                     (1.0,   10.0),   # raw RDKit SA score (1 = easy, 10 = hard)
    'logp':                         (-5.0,  10.0),
    'mol_weight':                   (0.0,  600.0),   # Da
    'tpsa':                         (0.0,  200.0),   # Å²
    'docking_score':                (-15.0,  0.0),   # kcal/mol (more negative = tighter binding)
    'lipinski_violations':          (0.0,    4.0),
    'admet_pass':                   (0.0,    1.0),
    # Drug range-distance objectives (0 when inside target range, else distance)
    'logp_range_distance':          (0.0,   15.0),
    'mol_weight_range_distance':    (0.0,  500.0),
    # ── NLO / quantum chemistry ────────────────────────────────────────────
    'homo_lumo_gap':                (0.0,   30.0),   # eV
    'homo_energy':                  (-15.0, -3.0),   # eV
    'lumo_energy':                  (-5.0,   5.0),   # eV
    'alpha_mean':                   (0.0,  300.0),   # Bohr³
    'beta_mean':                    (0.0,  500.0),   # a.u.
    'gamma':                        (0.0, 1000.0),   # a.u.
    'dipole_moment':                (0.0,   20.0),   # Debye
    'total_energy':                 (-5000.0, 0.0),  # Hartree (highly size-dependent)
    # NLO derived objectives
    'beta_gamma_ratio':             (0.0, 1000.0),   # beta/gamma (capped ±1000 in QCI)
    'total_energy_atom_ratio':      (-100.0,  0.0),  # total_energy / natoms  (Ha/atom)
    'alpha_range_distance':         (0.0,  500.0),   # 0 if alpha ∈ [100,500], else distance
    'homo_lumo_gap_range_distance': (0.0,   30.0),   # 0 if gap ∈ [2.5,3.5] eV, else distance
    # ── Structural archive measures ────────────────────────────────────────
    'num_atoms':                    (5.0,   35.0),
    'num_bonds':                    (4.0,   40.0),
    # ── PMO benchmark oracle scores ────────────────────────────────────────
    # Most TDC/PMO oracles return [0, 1]; special cases noted.
    # Ref: Gao et al. 2022 "Sample Efficiency Matters" (arXiv:2206.12411)
    'penalized_logp':               (-50.0, 30.0),   # unbounded; top solutions ~15–30
    'mw':                           (0.0, 1000.0),   # molecular weight oracle score
    'ring_count':                   (0.0,   10.0),   # number of rings
    'jnk3':                         (0.0,    1.0),
    'gsk3b':                        (0.0,    1.0),
    'drd2':                         (0.0,    1.0),
    'celecoxib_rediscovery':        (0.0,    1.0),
    'troglitazone_rediscovery':     (0.0,    1.0),
    'thiothixene_rediscovery':      (0.0,    1.0),
    'albuterol_similarity':         (0.0,    1.0),
    'mestranol_similarity':         (0.0,    1.0),
    'isomers_c7h8n2o2':             (0.0,    1.0),
    'isomers_c9h10n2o2pf2cl':       (0.0,    1.0),
    'median1':                      (0.0,    1.0),
    'median2':                      (0.0,    1.0),
    'osimertinib_mpo':              (0.0,    1.0),
    'fexofenadine_mpo':             (0.0,    1.0),
    'ranolazine_mpo':               (0.0,    1.0),
    'perindopril_mpo':              (0.0,    1.0),
    'amlodipine_mpo':               (0.0,    1.0),
    'sitagliptin_mpo':              (0.0,    1.0),
    'zaleplon_mpo':                 (0.0,    1.0),
    'valsartan_smarts':             (0.0,    1.0),
    'scaffold_hop':                 (0.0,    1.0),
    'deco_hop':                     (0.0,    1.0),
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ObjectiveSpec:
    """Specification for a single optimisation objective."""
    key: str                        # Property name returned by the oracle
    direction: str                  # 'maximize' | 'minimize'
    bounds: Tuple[float, float]     # (min, max) physical range for normalisation/display
    reference_value: float          # Worst-case scalar used in the hypervolume reference point


@dataclasses.dataclass
class MeasureSpec:
    """Specification for a single archive behavioural descriptor."""
    key: str                        # Property name returned by the oracle
    bounds: Tuple[float, float]     # (min, max) — values are clipped to this range before CVT lookup


# ---------------------------------------------------------------------------
# ProblemConfig
# ---------------------------------------------------------------------------

class ProblemConfig:
    """
    Complete configuration for a molecular optimisation problem domain.

    Owns objectives (with directions, bounds, reference values) and archive
    measures (with bounds).  All mutation methods return *new* instances so
    the original is never modified — safe to keep a base preset and derive
    experimental variants from it.
    """

    def __init__(
        self,
        name: str,
        description: str,
        atom_set: str,
        fitness_mode: str,
        objectives: List[ObjectiveSpec],
        measures: List[MeasureSpec],
    ) -> None:
        self.name = name
        self.description = description
        self.atom_set = atom_set          # 'nlo' | 'drug'
        self.fitness_mode = fitness_mode  # 'qc' | 'smartcadd' | 'rdkit'
        self.objectives = list(objectives)
        self.measures = list(measures)

    # ------------------------------------------------------------------
    # Derived properties — consumed directly by archives and main.py
    # ------------------------------------------------------------------

    @property
    def objective_keys(self) -> List[str]:
        return [o.key for o in self.objectives]

    @property
    def optimize_directions(self) -> List[Tuple[str, None]]:
        """List of ('max'|'min', None) tuples — format CVT archives expect."""
        _map = {'maximize': 'max', 'minimize': 'min'}
        return [(_map[o.direction], None) for o in self.objectives]

    @property
    def optimize_strings(self) -> List[str]:
        """List of 'maximize'|'minimize' strings — format CLI args use."""
        return [o.direction for o in self.objectives]

    @property
    def reference_point(self) -> List[float]:
        return [o.reference_value for o in self.objectives]

    @property
    def objective_bounds(self) -> List[Tuple[float, float]]:
        return [o.bounds for o in self.objectives]

    @property
    def measure_bounds(self) -> List[Tuple[float, float]]:
        return [m.bounds for m in self.measures]

    @property
    def measure_bounds_flat(self) -> List[float]:
        """Flattened [lo1, hi1, lo2, hi2, ...] — matches --measure-bounds CLI format."""
        flat: List[float] = []
        for lo, hi in self.measure_bounds:
            flat.extend([lo, hi])
        return flat

    @property
    def measure_keys(self) -> List[str]:
        return [m.key for m in self.measures]

    # ------------------------------------------------------------------
    # Mutation helpers — all return new ProblemConfig instances
    # ------------------------------------------------------------------

    def with_objectives(self, *keys: str) -> 'ProblemConfig':
        """Return a copy keeping only the named objectives (in given order).

        Raises KeyError if a key is not present in the current objectives.
        """
        obj_map = {o.key: o for o in self.objectives}
        missing = [k for k in keys if k not in obj_map]
        if missing:
            raise KeyError(
                f"Objectives not found in preset '{self.name}': {missing}. "
                f"Available: {self.objective_keys}"
            )
        new = copy.copy(self)
        new.objectives = [obj_map[k] for k in keys]
        return new

    def with_measures(self, *specs: MeasureSpec) -> 'ProblemConfig':
        """Return a copy with the measure list replaced entirely."""
        new = copy.copy(self)
        new.measures = list(specs)
        return new

    def override_bounds(self, key: str, bounds: Tuple[float, float]) -> 'ProblemConfig':
        """Return a copy with updated physical bounds for a named objective or measure.

        Searches objectives first, then measures.  Raises KeyError if not found.
        """
        new = copy.copy(self)
        new.objectives = list(self.objectives)
        new.measures = list(self.measures)
        for i, o in enumerate(new.objectives):
            if o.key == key:
                new.objectives[i] = dataclasses.replace(o, bounds=bounds)
                return new
        for i, m in enumerate(new.measures):
            if m.key == key:
                new.measures[i] = dataclasses.replace(m, bounds=bounds)
                return new
        all_keys = self.objective_keys + self.measure_keys
        raise KeyError(
            f"Key '{key}' not found in preset '{self.name}'. "
            f"Available keys: {all_keys}"
        )

    def override_reference(self, key: str, value: float) -> 'ProblemConfig':
        """Return a copy with an updated reference_value for a named objective.

        Raises KeyError if the objective key is not present.
        """
        new = copy.copy(self)
        new.objectives = list(self.objectives)
        for i, o in enumerate(new.objectives):
            if o.key == key:
                new.objectives[i] = dataclasses.replace(o, reference_value=value)
                return new
        raise KeyError(
            f"Objective '{key}' not found in preset '{self.name}'. "
            f"Available: {self.objective_keys}"
        )

    def replace(self, **kwargs) -> 'ProblemConfig':
        """Return a copy with top-level fields replaced.

        Valid kwargs: name, description, atom_set, fitness_mode, objectives, measures.
        """
        new = copy.copy(self)
        for k, v in kwargs.items():
            if not hasattr(new, k):
                raise TypeError(f"ProblemConfig has no field '{k}'")
            setattr(new, k, v)
        return new

    def __repr__(self) -> str:
        obj_summary = ', '.join(
            f"{o.key}({'↑' if o.direction == 'maximize' else '↓'})"
            for o in self.objectives
        )
        return f"ProblemConfig('{self.name}', [{obj_summary}])"


# ---------------------------------------------------------------------------
# Preset registry
# ---------------------------------------------------------------------------

_PRESETS: Dict[str, ProblemConfig] = {}


def register_preset(name: str, config: ProblemConfig) -> None:
    """Register a ProblemConfig under a name.  Overwrites silently."""
    _PRESETS[name] = config


def get_preset(name: str) -> ProblemConfig:
    """Return a *deep copy* of the named preset so mutations never affect the registry.

    Raises KeyError with a helpful message if not found.
    """
    if name not in _PRESETS:
        available = ', '.join(sorted(_PRESETS.keys()))
        raise KeyError(
            f"Unknown problem preset '{name}'. "
            f"Available: {available}"
        )
    return copy.deepcopy(_PRESETS[name])


def list_presets() -> List[str]:
    """Return a sorted list of registered preset names."""
    return sorted(_PRESETS.keys())


# ---------------------------------------------------------------------------
# bounds_for_keys: replaces the private _property_bounds_for_keys() that
# was duplicated in mome/main.py and mo_cma_mae/main.py.
# ---------------------------------------------------------------------------

def bounds_for_keys(
    keys: List[str],
    measure_bounds_flat: Optional[List[float]] = None,
) -> List[Tuple[float, float]]:
    """Return (min, max) pairs for a list of property keys.

    Resolution order:
      1. measure_bounds_flat (--measure-bounds) if provided and long enough
      2. PROPERTY_BOUNDS lookup
      3. Default (0.0, 1.0)
    """
    if measure_bounds_flat and len(measure_bounds_flat) >= 2 * len(keys):
        pairs = list(zip(measure_bounds_flat[::2], measure_bounds_flat[1::2]))
        return [(float(lo), float(hi)) for lo, hi in pairs[:len(keys)]]
    return [PROPERTY_BOUNDS.get(k, (0.0, 1.0)) for k in keys]


# ---------------------------------------------------------------------------
# resolve_from_args: fills an argparse Namespace from a named preset
# ---------------------------------------------------------------------------

def resolve_from_args(args) -> Optional[ProblemConfig]:
    """Return a ProblemConfig built from a parsed argparse Namespace.

    Returns None if args.problem is not set (backward-compatible: all existing
    run scripts that don't use --problem continue to work unchanged).

    The preset provides defaults; explicit CLI flags always win:
      --objectives     subsets/reorders the preset's objectives
      --optimize       overrides direction strings per objective
      --reference-point overrides reference values
      --measure-bounds  overrides measure bounds
    """
    problem_name = getattr(args, 'problem', None)
    if not problem_name:
        return None

    cfg = get_preset(problem_name)

    # --objectives: subset (and reorder) preset objectives, or add new ones
    objectives = getattr(args, 'objectives', None)
    if objectives:
        obj_map = {o.key: o for o in cfg.objectives}
        new_objs = []
        for k in objectives:
            if k in obj_map:
                new_objs.append(obj_map[k])
            else:
                # Key not in preset — build a default spec from PROPERTY_BOUNDS
                lo, hi = PROPERTY_BOUNDS.get(k, (0.0, 1.0))
                new_objs.append(ObjectiveSpec(k, 'maximize', (lo, hi), 0.0))
        cfg = cfg.replace(objectives=new_objs)

    # --optimize: override directions (must match length of objectives)
    optimize = getattr(args, 'optimize', None)
    if optimize and len(optimize) == len(cfg.objectives):
        new_objs = [
            dataclasses.replace(spec, direction=direction)
            for spec, direction in zip(cfg.objectives, optimize)
        ]
        cfg = cfg.replace(objectives=new_objs)

    # --reference-point: override reference values
    reference_point = getattr(args, 'reference_point', None)
    if reference_point and len(reference_point) == len(cfg.objectives):
        new_objs = [
            dataclasses.replace(spec, reference_value=float(val))
            for spec, val in zip(cfg.objectives, reference_point)
        ]
        cfg = cfg.replace(objectives=new_objs)

    # --measure-bounds: override measure bounds
    measure_bounds = getattr(args, 'measure_bounds', None)
    if measure_bounds and len(measure_bounds) >= 2 * len(cfg.measures):
        pairs = list(zip(measure_bounds[::2], measure_bounds[1::2]))
        new_measures = [
            dataclasses.replace(m, bounds=(float(lo), float(hi)))
            for m, (lo, hi) in zip(cfg.measures, pairs)
        ]
        cfg = cfg.replace(measures=new_measures)

    return cfg


# ---------------------------------------------------------------------------
# Built-in presets
# ---------------------------------------------------------------------------

_STRUCTURAL_MEASURES = [
    MeasureSpec('num_atoms', (5.0, 35.0)),
    MeasureSpec('num_bonds', (4.0, 40.0)),
]


def _obj(key: str, direction: str, ref: float) -> ObjectiveSpec:
    """Shorthand: build ObjectiveSpec from PROPERTY_BOUNDS."""
    bounds = PROPERTY_BOUNDS.get(key, (0.0, 1.0))
    return ObjectiveSpec(key, direction, bounds, ref)


# ── NLO presets ──────────────────────────────────────────────────────────────

_nlo_cfg = ProblemConfig(
    name='nlo',
    description=(
        'NLO 4-objective: maximise beta/gamma ratio, minimise energy/atom ratio, '
        'minimise alpha range distance, minimise HOMO-LUMO gap range distance. '
        'Reference point matches the established run-script default [0,0,500,100].'
    ),
    atom_set='nlo',
    fitness_mode='qc',
    objectives=[
        _obj('beta_gamma_ratio',             'maximize',   0.0),
        _obj('total_energy_atom_ratio',      'minimize',   0.0),
        _obj('alpha_range_distance',         'minimize', 500.0),
        _obj('homo_lumo_gap_range_distance', 'minimize', 100.0),
    ],
    measures=list(_STRUCTURAL_MEASURES),
)
register_preset('nlo', _nlo_cfg)
register_preset('nlo_4obj', _nlo_cfg)  # backward-compat alias

register_preset('nlo_2obj', ProblemConfig(
    name='nlo_2obj',
    description='NLO 2-objective: maximise beta_mean and homo_lumo_gap.',
    atom_set='nlo',
    fitness_mode='qc',
    objectives=[
        _obj('beta_mean',     'maximize', 0.0),
        _obj('homo_lumo_gap', 'maximize', 0.0),
    ],
    measures=list(_STRUCTURAL_MEASURES),
))

register_preset('nlo_1obj_beta', ProblemConfig(
    name='nlo_1obj_beta',
    description='NLO single-objective: maximise beta_mean.',
    atom_set='nlo',
    fitness_mode='qc',
    objectives=[
        _obj('beta_mean', 'maximize', 0.0),
    ],
    measures=list(_STRUCTURAL_MEASURES),
))

# ── Drug / SmartCADD presets ──────────────────────────────────────────────────

register_preset('drug_4obj', ProblemConfig(
    name='drug_4obj',
    description=(
        'Drug 4-objective: maximise QED + SA (inverted), minimise logP range '
        'distance + mol-weight range distance. Matches DrugOracle formulation.'
    ),
    atom_set='drug',
    fitness_mode='rdkit',
    objectives=[
        _obj('qed',                      'maximize',   0.0),
        _obj('sa',                       'maximize',   0.0),
        _obj('logp_range_distance',      'minimize',  15.0),
        _obj('mol_weight_range_distance','minimize', 500.0),
    ],
    measures=list(_STRUCTURAL_MEASURES),
))

register_preset('drug_2obj', ProblemConfig(
    name='drug_2obj',
    description='Drug 2-objective: maximise QED and SA (inverted SA score).',
    atom_set='drug',
    fitness_mode='rdkit',
    objectives=[
        _obj('qed', 'maximize', 0.0),
        _obj('sa',  'maximize', 0.0),
    ],
    measures=list(_STRUCTURAL_MEASURES),
))

register_preset('drug_1obj_qed', ProblemConfig(
    name='drug_1obj_qed',
    description='Drug single-objective: maximise QED.',
    atom_set='drug',
    fitness_mode='rdkit',
    objectives=[
        _obj('qed', 'maximize', 0.0),
    ],
    measures=list(_STRUCTURAL_MEASURES),
))

register_preset('drug_1obj_docking', ProblemConfig(
    name='drug_1obj_docking',
    description='Drug single-objective: minimise docking score via SmartCADD pipeline '
                '(ADMET pre-filter → Smina docking). Lower kcal/mol = tighter binding.',
    atom_set='drug',
    fitness_mode='smartcadd',
    objectives=[
        # docking_score is in kcal/mol; 0.0 = no binding (worst), −15 = very tight (best)
        _obj('docking_score', 'minimize', 0.0),
    ],
    measures=list(_STRUCTURAL_MEASURES),
))

register_preset('drug_smartcadd_2obj', ProblemConfig(
    name='drug_smartcadd_2obj',
    description='Drug 2-objective via SmartCADD descriptors: maximise QED and SA.',
    atom_set='drug',
    fitness_mode='smartcadd',
    objectives=[
        _obj('qed', 'maximize', 0.0),
        _obj('sa',  'maximize', 0.0),
    ],
    measures=list(_STRUCTURAL_MEASURES),
))

# ── PMO benchmark presets ─────────────────────────────────────────────────────
# Configured to match Gao et al. 2022 "Sample Efficiency Matters: A Benchmark
# for Practical Molecular Optimization" (arXiv:2206.12411).
#
# Standard PMO protocol:
#   - Budget: 10,000 oracle calls per run
#   - Evaluation: AUC top-1 / top-10 / top-100 over the budget
#   - atom_set: 'drug'  (C/N/O/S/F/Cl/Br)
#   - All oracles: single-objective, maximize
#
# Usage:
#   --problem pmo_qed         # local oracle, no extra deps
#   --problem pmo_jnk3        # requires PyTDC (pip install PyTDC)
#
# The oracle is passed separately to the algorithm via --oracle <name>.
# These presets fix the bounds and reference value for each oracle so
# archives and hypervolume calculations are consistently configured.

# Shared placeholder kept for backward compat
register_preset('pmo', ProblemConfig(
    name='pmo',
    description=(
        'PMO benchmark placeholder — atom_set and fitness_mode only. '
        'Use a specific pmo_<oracle> preset for configured bounds. '
        'Objectives are set dynamically by the PMO oracle (--oracle flag).'
    ),
    atom_set='drug',
    fitness_mode='rdkit',
    objectives=[],  # filled dynamically per oracle run
    measures=list(_STRUCTURAL_MEASURES),
))


def _pmo(oracle_key: str, description: str,
         lo: float = 0.0, hi: float = 1.0, ref: float = 0.0) -> ProblemConfig:
    """Shorthand for building a single-objective PMO preset."""
    return ProblemConfig(
        name=f'pmo_{oracle_key}',
        description=description,
        atom_set='drug',
        fitness_mode='rdkit',
        objectives=[ObjectiveSpec(oracle_key, 'maximize', (lo, hi), ref)],
        measures=list(_STRUCTURAL_MEASURES),
    )


# ── Local oracles (no extra deps) ─────────────────────────────────────────────

register_preset('pmo_qed', _pmo(
    'qed', 'PMO: Quantitative Estimate of Drug-likeness [0,1]. Gao et al. 2022.',
))
register_preset('pmo_sa', _pmo(
    'sa', 'PMO: Synthetic Accessibility score, inverted to [0,1] (1=easy). Gao et al. 2022.',
))
register_preset('pmo_penalized_logp', _pmo(
    'penalized_logp',
    'PMO: Penalized logP (logP minus SA and ring penalties). Unbounded; top solutions ~15–30. '
    'Gao et al. 2022.',
    lo=-50.0, hi=30.0, ref=-50.0,
))
register_preset('pmo_logp', _pmo(
    'logp', 'PMO: Raw Wildman-Crippen logP. [−5, 10] typical range. Gao et al. 2022.',
    lo=-5.0, hi=10.0, ref=-5.0,
))

# ── Kinase / receptor activity (TDC) ─────────────────────────────────────────

register_preset('pmo_jnk3', _pmo(
    'jnk3',
    'PMO: JNK3 kinase inhibitor activity, predicted by RF model [0,1]. Gao et al. 2022.',
))
register_preset('pmo_gsk3b', _pmo(
    'gsk3b',
    'PMO: GSK3β kinase inhibitor activity, predicted by RF model [0,1]. Gao et al. 2022.',
))
register_preset('pmo_drd2', _pmo(
    'drd2',
    'PMO: DRD2 dopamine receptor activity, predicted by REINVENT model [0,1]. Gao et al. 2022.',
))

# ── MPO (multi-property, combined to single [0,1] score) ─────────────────────

register_preset('pmo_osimertinib_mpo', _pmo(
    'osimertinib_mpo',
    'PMO: Osimertinib MPO — EGFR inhibitor property profile [0,1]. Gao et al. 2022.',
))
register_preset('pmo_fexofenadine_mpo', _pmo(
    'fexofenadine_mpo',
    'PMO: Fexofenadine MPO — antihistamine property profile [0,1]. Gao et al. 2022.',
))
register_preset('pmo_ranolazine_mpo', _pmo(
    'ranolazine_mpo',
    'PMO: Ranolazine MPO — anti-angina property profile [0,1]. Gao et al. 2022.',
))
register_preset('pmo_perindopril_mpo', _pmo(
    'perindopril_mpo',
    'PMO: Perindopril MPO — ACE inhibitor property profile [0,1]. Gao et al. 2022.',
))
register_preset('pmo_amlodipine_mpo', _pmo(
    'amlodipine_mpo',
    'PMO: Amlodipine MPO — calcium channel blocker property profile [0,1]. Gao et al. 2022.',
))
register_preset('pmo_sitagliptin_mpo', _pmo(
    'sitagliptin_mpo',
    'PMO: Sitagliptin MPO — DPP4 inhibitor property profile [0,1]. Gao et al. 2022.',
))
register_preset('pmo_zaleplon_mpo', _pmo(
    'zaleplon_mpo',
    'PMO: Zaleplon MPO — GABA_A receptor property profile [0,1]. Gao et al. 2022.',
))

# ── Structural similarity / rediscovery ───────────────────────────────────────

register_preset('pmo_celecoxib_rediscovery', _pmo(
    'celecoxib_rediscovery',
    'PMO: Celecoxib rediscovery — Tanimoto similarity to celecoxib [0,1]. Gao et al. 2022.',
))
register_preset('pmo_troglitazone_rediscovery', _pmo(
    'troglitazone_rediscovery',
    'PMO: Troglitazone rediscovery — Tanimoto similarity to troglitazone [0,1]. Gao et al. 2022.',
))
register_preset('pmo_thiothixene_rediscovery', _pmo(
    'thiothixene_rediscovery',
    'PMO: Thiothixene rediscovery — Tanimoto similarity to thiothixene [0,1]. Gao et al. 2022.',
))
register_preset('pmo_albuterol_similarity', _pmo(
    'albuterol_similarity',
    'PMO: Albuterol similarity — Tanimoto similarity to albuterol [0,1]. Gao et al. 2022.',
))
register_preset('pmo_mestranol_similarity', _pmo(
    'mestranol_similarity',
    'PMO: Mestranol similarity — Tanimoto similarity to mestranol [0,1]. Gao et al. 2022.',
))

# ── Isomer design ─────────────────────────────────────────────────────────────

register_preset('pmo_isomers_c7h8n2o2', _pmo(
    'isomers_c7h8n2o2',
    'PMO: Isomers C7H8N2O2 — score for matching molecular formula [0,1]. Gao et al. 2022.',
))
register_preset('pmo_isomers_c9h10n2o2pf2cl', _pmo(
    'isomers_c9h10n2o2pf2cl',
    'PMO: Isomers C9H10N2O2PF2Cl — score for matching molecular formula [0,1]. Gao et al. 2022.',
))

# ── Median molecule design ────────────────────────────────────────────────────

register_preset('pmo_median1', _pmo(
    'median1',
    'PMO: Median molecules 1 — balanced similarity to two reference structures [0,1]. '
    'Gao et al. 2022.',
))
register_preset('pmo_median2', _pmo(
    'median2',
    'PMO: Median molecules 2 — balanced similarity to two reference structures [0,1]. '
    'Gao et al. 2022.',
))

# ── Scaffold / decoration hopping ─────────────────────────────────────────────

register_preset('pmo_scaffold_hop', _pmo(
    'scaffold_hop',
    'PMO: Scaffold hopping — scaffold-matched similarity score [0,1]. Gao et al. 2022.',
))
register_preset('pmo_deco_hop', _pmo(
    'deco_hop',
    'PMO: Decoration hopping — decoration-matched similarity score [0,1]. Gao et al. 2022.',
))

# ── SMARTS filter ─────────────────────────────────────────────────────────────

register_preset('pmo_valsartan_smarts', _pmo(
    'valsartan_smarts',
    'PMO: Valsartan SMARTS — combined SMARTS + activity score [0,1]. Gao et al. 2022.',
))
