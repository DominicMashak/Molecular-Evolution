#!/usr/bin/env python3
"""
MOEA/D Molecular Optimisation using pymoo (ParallelMOEAD).

Evolves molecules in the transformer UMAP embedding space.  MOEA/D decomposes
the multi-objective problem into scalar subproblems via weight vectors and a
scalarization function (Tchebicheff / PBI / WeightedSum).  Neighbourhood-based
parent selection drives exploitation of neighbouring subproblems.

Uses ParallelMOEAD (batch mode) so each generation evaluates pop_size offspring,
matching the NSGA-III generation structure.  Supports SMILES, SELFIES, BigSMILES,
and SLICES encodings with ChemBERTa / PolyBERT / MatText embedders.

Examples
--------
# Drug design, 2 objectives — auto TCH decomposition
python main.py --fitness-mode smartcadd --encoding smiles \\
               --atom-set drug \\
               --objectives qed sa_score \\
               --optimize maximize minimize \\
               --reference-point 0.0 10.0 \\
               --n_gen 300 --output_dir results_drug_moead

# NLO, 4 objectives — auto PBI decomposition
python main.py --calculator dft --functional HF --basis 3-21G \\
               --atom-set nlo \\
               --objectives beta_gamma_ratio total_energy_atom_ratio \\
                            alpha_range_distance homo_lumo_gap_range_distance \\
               --optimize maximize minimize minimize minimize \\
               --reference-point 0.0 0.0 500.0 100.0 \\
               --n_gen 500 --output_dir results_nlo_moead

# 8-objective drug design — explicit TCH (recommended for many objectives)
python main.py --fitness-mode smartcadd --encoding smiles \\
               --objectives qed sa_score docking_score mol_weight_range_distance \\
                            logp_range_distance tpsa_range_distance \\
                            lipinski_violations admet_pass \\
               --optimize maximize minimize minimize minimize \\
                          minimize minimize minimize maximize \\
               --decomposition tchebicheff \\
               --n-partitions 3 \\
               --n_gen 500 --output_dir results_8obj_moead
"""

import sys
import os
import argparse
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'molev_utils')))

import numpy as np
import optimizer as op
from molecule_generator import MoleculeGenerator
from quantum_chemistry_interface import QuantumChemistryInterface


# ---------------------------------------------------------------------------
# Auto n_partitions schedule (same as NSGA-III)
# ---------------------------------------------------------------------------

_PARTITIONS_SCHEDULE: dict = {2: 12, 3: 8, 4: 6, 5: 5, 6: 4, 7: 4, 8: 3}


def _get_n_partitions(n_obj: int, override=None) -> int:
    if override is not None:
        return override
    return _PARTITIONS_SCHEDULE.get(n_obj, max(2, 12 - n_obj))


# ---------------------------------------------------------------------------
# Objective parsing (mirrors nsga2/main.py and nsga3/main.py)
# ---------------------------------------------------------------------------

def parse_optimize_arg(arg: str):
    if arg.lower() in ('maximize', 'max'):
        return ('max', None)
    elif arg.lower() in ('minimize', 'min'):
        return ('min', None)
    elif arg.startswith('target:') or arg.startswith('t:'):
        try:
            _, value = arg.split(':', 1)
            return ('target', float(value))
        except ValueError:
            raise ValueError(f"Invalid target format: {arg}")
    else:
        raise ValueError(f"Unknown optimize type: {arg!r}. "
                         "Use 'maximize', 'minimize', or 'target:X'.")


# ---------------------------------------------------------------------------
# Measure bounds parser (verbatim from cma_mae/main.py)
# ---------------------------------------------------------------------------

def _parse_measure_bounds(measure_bounds_flat, measure_keys):
    defaults = {'num_atoms': (1.0, 50.0), 'num_bonds': (0.0, 60.0)}
    if measure_bounds_flat:
        pairs = list(zip(measure_bounds_flat[::2], measure_bounds_flat[1::2]))
        return [(float(lo), float(hi)) for lo, hi in pairs]
    return [defaults.get(k, (0.0, 1.0)) for k in measure_keys]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="MOEA/D Molecular Optimisation (pymoo ParallelMOEAD, "
                    "latent-space genome)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Calculator options ─────────────────────────────────────────────────
    parser.add_argument('--calculator', type=str, default=None,
                        choices=['dft', 'cc', 'semiempirical', 'xtb'],
                        help='Calculator type (required for --fitness-mode qc)')
    parser.add_argument('--basis', type=str, default="6-31G")
    parser.add_argument('--functional', type=str, default="B3LYP")
    parser.add_argument('--method', type=str, default="full_tensor")
    parser.add_argument('--se-method', type=str, default="PM7")
    parser.add_argument('--xtb-method', type=str, default="GFN2-xTB")
    parser.add_argument('--field-strength', type=float, default=0.001)

    # ── Population / generation ────────────────────────────────────────────
    parser.add_argument('--pop_size', type=int, default=None,
                        help='MOEA/D population size (= number of weight vectors). '
                             'Defaults to number of Das-Dennis reference directions.')
    parser.add_argument('--n_gen', type=int, default=300)
    parser.add_argument('--log_frequency', type=int, default=10)
    parser.add_argument('--save_frequency', type=int, default=50)
    parser.add_argument('--output_dir', type=str, default="moead_results")

    # ── Fitness mode ───────────────────────────────────────────────────────
    parser.add_argument('--fitness-mode', type=str, default='qc',
                        choices=['qc', 'smartcadd'])
    parser.add_argument('--smartcadd-path', type=str, default=None)
    parser.add_argument('--smartcadd-mode', type=str, default='descriptors',
                        choices=['descriptors', 'docking'])
    parser.add_argument('--protein-code', type=str, default=None)
    parser.add_argument('--protein-path', type=str, default=None)
    parser.add_argument('--alert-collection', type=str, default=None)

    # ── Molecule encoding ──────────────────────────────────────────────────
    parser.add_argument('--atom-set', type=str, default=None,
                        choices=['nlo', 'drug'])
    parser.add_argument('--encoding', type=str, default='smiles',
                        choices=['smiles', 'selfies', 'bigsmiles', 'slices'])
    parser.add_argument('--bigsmiles-n-samples', type=int, default=3)
    parser.add_argument('--bigsmiles-dp', type=int, default=None)
    parser.add_argument('--crystal-element-set', type=str, default='oxides',
                        choices=['oxides', 'semiconductor', 'semiconductors',
                                 'halide_perovskite'])
    parser.add_argument('--crystal-hardness', action='store_true')
    parser.add_argument('--crystal-phonon-stability', action='store_true')
    parser.add_argument('--crystal-mobility', action='store_true')
    parser.add_argument('--crystal-hull-distance', action='store_true')

    # ── Objectives ────────────────────────────────────────────────────────
    parser.add_argument('--objectives', type=str, nargs='+', required=True,
                        help='Objective keys (e.g. --objectives qed sa_score)')
    parser.add_argument('--optimize', type=str, nargs='+', required=True,
                        help='Direction per objective: maximize, minimize, or target:X')
    parser.add_argument('--reference-point', type=float, nargs='+', default=None,
                        help='HV reference in original units, one per objective. '
                             'Should be the worst-case value for each objective.')

    # ── MOEA/D specific ───────────────────────────────────────────────────
    parser.add_argument('--decomposition', type=str, default=None,
                        choices=['tchebicheff', 'tch', 'pbi', 'weighted_sum', 'ws'],
                        help='Scalarization function. Auto: TCH for n_obj≤2, '
                             'PBI for n_obj>2.')
    parser.add_argument('--n-neighbors', type=int, default=20,
                        help='Neighbourhood size for weight-vector mating (default: 20)')
    parser.add_argument('--prob-neighbor-mating', type=float, default=0.9,
                        help='Probability of selecting parents from neighbourhood '
                             'vs. whole population (default: 0.9)')
    parser.add_argument('--n-partitions', type=int, default=None,
                        help='Das-Dennis lattice partitions. Auto-scheduled from '
                             'n_obj: {2:12, 3:8, 4:6, 5:5, 6:4, 7:4, 8:3}')
    parser.add_argument('--sbx-eta', type=float, default=20.0,
                        help='SBX crossover distribution index (default: 20)')
    parser.add_argument('--sbx-prob', type=float, default=1.0,
                        help='SBX crossover probability per variable (default: 1.0)')
    parser.add_argument('--pm-eta', type=float, default=20.0,
                        help='Polynomial mutation distribution index (default: 20)')
    parser.add_argument('--pm-prob', type=float, default=None,
                        help='PM mutation probability per variable. '
                             'Defaults to 1/embedding_dims.')
    parser.add_argument('--pool-max-size', type=int, default=10000)

    # ── Embedding ──────────────────────────────────────────────────────────
    parser.add_argument('--embedding-model', type=str,
                        default='DeepChem/ChemBERTa-77M-MTR')
    parser.add_argument('--embedding-dims', type=int, default=10)
    parser.add_argument('--embedding-device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda', 'mps'])
    parser.add_argument('--embedding-sample-size', type=int, default=10000)

    # ── Domain config / problem preset ────────────────────────────────────
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--problem', type=str, default=None)

    # ── Other ──────────────────────────────────────────────────────────────
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--verbose', action='store_true')

    # ── Load domain config ─────────────────────────────────────────────────
    _pre = argparse.ArgumentParser(add_help=False)
    _pre.add_argument('--config', type=str, default=None)
    _known, _ = _pre.parse_known_args()
    if _known.config:
        from config_loader import inject_config_defaults
        inject_config_defaults(parser, _known.config)

    args = parser.parse_args()

    from problem_config import resolve_from_args
    _problem = resolve_from_args(args)
    if _problem is not None and args.reference_point is None:
        args.reference_point = _problem.reference_point

    # ── Validate objectives ────────────────────────────────────────────────
    if len(args.objectives) != len(args.optimize):
        parser.error(
            f"--objectives has {len(args.objectives)} entries but "
            f"--optimize has {len(args.optimize)}. Must match."
        )

    optimize_parsed = []
    for opt_str in args.optimize:
        try:
            optimize_parsed.append(parse_optimize_arg(opt_str))
        except ValueError as e:
            parser.error(str(e))

    negate_mask = [mode == 'max' for mode, _ in optimize_parsed]

    for mode, _ in optimize_parsed:
        if mode == 'target':
            parser.error(
                "'target:X' is not yet supported. Pre-compute a range-distance "
                "objective and use 'minimize' instead."
            )

    # ── Seed ──────────────────────────────────────────────────────────────
    random.seed(args.seed)
    np.random.seed(args.seed)
    from rdkit import rdBase
    rdBase.SeedRandomNumberGenerator(args.seed)

    # ── Crystal (SLICES) setup ────────────────────────────────────────────
    crystal_eval = None
    if args.encoding == 'slices':
        from crystal_qe_interface import CrystalQEInterface
        crystal_eval = CrystalQEInterface(
            verbose=args.verbose,
            calc_hardness=args.crystal_hardness,
            calc_phonon_stability=args.crystal_phonon_stability,
            calc_mobility=args.crystal_mobility,
            calc_hull_distance=args.crystal_hull_distance,
        )

    # ── Atom set ──────────────────────────────────────────────────────────
    if args.encoding == 'slices':
        atom_set = getattr(args, 'crystal_element_set', 'oxides')
    elif args.atom_set:
        atom_set = args.atom_set
    elif args.fitness_mode == 'smartcadd':
        atom_set = 'drug'
    else:
        atom_set = 'nlo'

    # ── Evaluation interface ───────────────────────────────────────────────
    if args.encoding == 'slices':
        eval_interface = crystal_eval
    elif args.fitness_mode == 'smartcadd':
        from smartcadd_interface import SmartCADDInterface
        smartcadd_kwargs = {'mode': args.smartcadd_mode}
        if args.smartcadd_path:
            smartcadd_kwargs['smartcadd_path'] = args.smartcadd_path
        if args.protein_code:
            smartcadd_kwargs['protein_code'] = args.protein_code
        if args.protein_path:
            smartcadd_kwargs['protein_path'] = args.protein_path
        if args.alert_collection:
            smartcadd_kwargs['alert_collection_path'] = args.alert_collection
        eval_interface = SmartCADDInterface(verbose=args.verbose, **smartcadd_kwargs)
    else:
        if args.calculator is None:
            parser.error("--calculator is required when --fitness-mode qc")
        calculator_kwargs = {}
        if args.basis:
            calculator_kwargs['basis'] = args.basis
        if args.functional:
            calculator_kwargs['functional'] = args.functional
        if args.se_method:
            calculator_kwargs['se_method'] = args.se_method
        if args.xtb_method:
            calculator_kwargs['xtb_method'] = args.xtb_method
        eval_interface = QuantumChemistryInterface(
            calculator_type=args.calculator,
            calculator_kwargs=calculator_kwargs,
            method=args.method,
            field_strength=args.field_strength,
            verbose=args.verbose,
        )

    generator = MoleculeGenerator(
        seed=args.seed,
        atom_set=atom_set,
        encoding=args.encoding,
        n_samples=getattr(args, 'bigsmiles_n_samples', 3),
        dp=getattr(args, 'bigsmiles_dp', None),
        element_set=getattr(args, 'crystal_element_set', 'oxides'),
    )

    # ── Embedder setup ────────────────────────────────────────────────────
    from molecular_embedder import MolecularEmbedder
    _emb_model = args.embedding_model
    _emb_tok = None
    _emb_input_fmt = 'smiles'
    if args.encoding == 'bigsmiles' and _emb_model == 'DeepChem/ChemBERTa-77M-MTR':
        _emb_model = 'kuelumbus/polyBERT'
        _emb_input_fmt = 'bigsmiles'
    elif args.encoding == 'slices' and _emb_model == 'DeepChem/ChemBERTa-77M-MTR':
        _emb_model = 'n0w0f/MatText-slices-2m'
        _emb_tok = 'bert-base-uncased'
        _emb_input_fmt = 'slices'

    print(f"Fitting {_emb_model} embedder on {args.embedding_sample_size} structures...")
    _emb_raw = generator.generate_initial_population(args.embedding_sample_size)
    if args.encoding == 'slices':
        _emb_strings = [s for s in _emb_raw if s is not None]
    else:
        _emb_strings = [generator.decode_to_smiles(s) for s in _emb_raw]
        _emb_strings = [s for s in _emb_strings if s is not None]

    embedder = MolecularEmbedder(
        model_name=_emb_model,
        n_components=args.embedding_dims,
        device=args.embedding_device,
        random_state=args.seed,
        input_format=_emb_input_fmt,
        tokenizer_name=_emb_tok,
    )
    embedder.fit(_emb_strings)
    print(f"Embedder fitted (n_components={args.embedding_dims}).")

    z_bounds = embedder.get_measure_bounds()

    # ── Build initial pool (free — cached embeddings from fit) ────────────
    print("Seeding z_smiles pool from embedder fitting sample...")
    initial_pool = []
    for smi in _emb_strings:
        try:
            z = embedder.embed(smi)
            initial_pool.append((z.astype(np.float64), smi))
        except Exception:
            pass
    print(f"Pool seeded with {len(initial_pool)} molecules.")

    # ── evaluate_fn (original units — no negation here) ───────────────────
    _genotype_key = 'slices' if args.encoding == 'slices' else 'smiles'

    def _error_props():
        base = {'smiles': None, 'num_atoms': 0.0, 'num_bonds': 0.0,
                'error': 'Invalid molecule'}
        for obj_key in args.objectives:
            base.setdefault(obj_key, 0.0)
        return base

    def evaluate_solution(smiles: str) -> dict:
        if args.encoding == 'slices':
            if smiles is None:
                return _error_props()
            result = eval_interface.calculate(smiles)
            props = dict(result)
            if embedder is not None:
                emb = embedder.embed(smiles)
                for i, val in enumerate(emb):
                    props[f'emb_{i}'] = float(val)
            return props

        from rdkit import Chem as _Chem
        if smiles is None:
            return _error_props()
        mol = _Chem.MolFromSmiles(smiles)
        if mol is None:
            return _error_props()
        num_atoms = mol.GetNumAtoms()
        num_bonds = mol.GetNumBonds()
        result = eval_interface.calculate(smiles)
        props = {
            'smiles': smiles,
            'num_atoms': float(num_atoms),
            'num_bonds': float(num_bonds),
            'num_atoms_bin': min(9, max(0, (num_atoms - 5) // 3)),
            'num_bonds_bin': min(9, max(0, (num_bonds - 5) // 3)),
            'error': result.get('error'),
        }
        for k, v in result.items():
            if k not in props and k != 'smiles':
                props[k] = v if v is not None else 0.0
        if embedder is not None:
            emb = embedder.embed(smiles)
            for i, val in enumerate(emb):
                props[f'emb_{i}'] = float(val)
        return props

    def generate_fn():
        pop = generator.generate_initial_population(1)
        if not pop:
            return None
        if args.encoding == 'slices':
            return pop[0]
        return generator.decode_to_smiles(pop[0])

    # ── pymoo MOEA/D setup ────────────────────────────────────────────────
    try:
        from pymoo.algorithms.moo.moead import ParallelMOEAD
        from pymoo.util.ref_dirs import get_reference_directions
        from pymoo.operators.crossover.sbx import SBX
        from pymoo.operators.mutation.pm import PM
    except ImportError:
        print("ERROR: pymoo is not installed. Run: pip install pymoo")
        sys.exit(1)

    n_obj = len(args.objectives)
    n_partitions = _get_n_partitions(n_obj, args.n_partitions)
    ref_dirs = get_reference_directions('das-dennis', n_obj, n_partitions=n_partitions)
    pop_size = args.pop_size if args.pop_size is not None else len(ref_dirs)

    if args.pop_size is not None and args.pop_size < len(ref_dirs):
        print(
            f"WARNING: --pop_size {args.pop_size} < n_ref_dirs {len(ref_dirs)}. "
            "MOEA/D performance may degrade. Consider omitting --pop_size."
        )

    # Decomposition function
    decomp = None
    if args.decomposition is not None:
        from pymoo.decomposition.tchebicheff import Tchebicheff
        from pymoo.decomposition.pbi import PBI
        from pymoo.decomposition.weighted_sum import WeightedSum
        _decomp_map = {
            'tchebicheff': Tchebicheff(),
            'tch':         Tchebicheff(),
            'pbi':         PBI(),
            'weighted_sum': WeightedSum(),
            'ws':           WeightedSum(),
        }
        decomp = _decomp_map[args.decomposition]
    # decomp=None → pymoo auto: TCH for n_obj≤2, PBI for n_obj>2

    pm_prob = args.pm_prob if args.pm_prob is not None else 1.0 / args.embedding_dims

    algorithm = ParallelMOEAD(
        ref_dirs=ref_dirs,
        decomposition=decomp,
        n_neighbors=args.n_neighbors,
        prob_neighbor_mating=args.prob_neighbor_mating,
        crossover=SBX(eta=args.sbx_eta, prob=args.sbx_prob),
        mutation=PM(eta=args.pm_eta, prob=pm_prob),
    )

    problem = op.MolecularProblem(
        n_obj=n_obj,
        embed_dim=args.embedding_dims,
        z_bounds=z_bounds,
    )

    algorithm.setup(problem, seed=args.seed, verbose=False)

    # ── HV reference point ────────────────────────────────────────────────
    hv_ref_point = None
    if args.reference_point is not None:
        if len(args.reference_point) != n_obj:
            parser.error(
                f"--reference-point has {len(args.reference_point)} values "
                f"but there are {n_obj} objectives."
            )
        hv_ref_point = np.array(args.reference_point, dtype=float)
    else:
        print("WARNING: --reference-point not provided. "
              "Hypervolume will be logged as 0.0.")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Print configuration ───────────────────────────────────────────────
    opt_labels = ['MAX' if n else 'MIN' for n in negate_mask]
    _decomp_label = args.decomposition or ('TCH (auto)' if n_obj <= 2 else 'PBI (auto)')
    _emb_label = 'MatText UMAP' if args.encoding == 'slices' else 'ChemBERTa UMAP'
    print(f"\nMOEA/D configuration:")
    print(f"  Objectives:      {list(zip(args.objectives, opt_labels))}")
    print(f"  Decomposition:   {_decomp_label}")
    print(f"  n_neighbors:     {args.n_neighbors}")
    print(f"  prob_mating:     {args.prob_neighbor_mating}")
    print(f"  Embed dim:       {args.embedding_dims} ({_emb_label})")
    print(f"  n_partitions:    {n_partitions}  →  {len(ref_dirs)} weight vectors")
    print(f"  pop_size:        {pop_size}")
    print(f"  SBX eta:         {args.sbx_eta}, prob: {args.sbx_prob}")
    print(f"  PM  eta:         {args.pm_eta},  prob: {pm_prob:.4f}")
    print(f"  HV ref:          {hv_ref_point}")
    print(f"  Encoding:        {args.encoding}")
    print(f"  Pool max:        {args.pool_max_size}")

    # ── Create optimizer and run ──────────────────────────────────────────
    optimizer = op.MOEADOptimizer(
        algorithm=algorithm,
        problem=problem,
        embedder=embedder,
        mutate_fn=generator.mutate_as_smiles,
        generate_fn=generate_fn,
        evaluate_fn=evaluate_solution,
        objective_keys=args.objectives,
        negate_mask=negate_mask,
        hv_ref_point=hv_ref_point,
        output_dir=args.output_dir,
        pool_max_size=args.pool_max_size,
        encoding=args.encoding,
        initial_pool=initial_pool,
    )

    optimizer.run(
        n_generations=args.n_gen,
        log_frequency=args.log_frequency,
        save_frequency=args.save_frequency,
    )

    best = optimizer.get_best_solution()
    if best:
        key0 = args.objectives[0]
        print(f"\nBest solution ({key0}): {best.get(_genotype_key)}")
        for obj_key in args.objectives:
            print(f"  {obj_key}: {best.get(obj_key)}")


if __name__ == "__main__":
    main()
