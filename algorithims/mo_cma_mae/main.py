import sys
import os
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'molev_utils')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'mome')))

import optimizer as op

from molecule_generator import MoleculeGenerator
from quantum_chemistry_interface import QuantumChemistryInterface


def main():
    parser = argparse.ArgumentParser(
        description="MO-CMA-MAE: Multi-Objective CMA-MAE Molecular Optimisation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
MO-CMA-MAE extends CMA-MAE to multi-objective optimisation.

Architecture:
  - Adaptation archive: pyribs CVTArchive (threshold annealing, learning_rate < 1)
    Stores latent vectors z; uses hypervolume contribution as scalar CMA-ES signal.
  - Result archive: CVTMOMEArchive (Pareto front per CVT cell)
    Stores full multi-objective solutions; reports MOQD and global hypervolume.

Both archives share the same CVT tessellation. CVT cells can be defined by:
  - structural measures (num_atoms, num_bonds)
  - embedding measures (ChemBERTa-2 + UMAP, same as CVT-MOME)

Examples:
  # NLO, 4 objectives, CVT + ChemBERTa embeddings (default)
  python main.py \\
    --fitness-mode qc --calculator dft --functional HF --basis 3-21G \\
    --atom-set nlo \\
    --cvt-measures embedding --n-centroids 100 --embedding-dims 10 \\
    --objectives beta_gamma_ratio total_energy_atom_ratio alpha_range_distance homo_lumo_gap_range_distance \\
    --optimize maximize minimize minimize minimize \\
    --reference-point 0.0 0.0 500.0 100.0 \\
    --n_gen 500 --pop_size 100 --output_dir results_mo_nlo

  # Drug-like, 2 objectives (QED + SA), CVT structural
  python main.py \\
    --fitness-mode smartcadd --smartcadd-mode descriptors --atom-set drug \\
    --cvt-measures structural --n-centroids 100 \\
    --objectives qed sa_score --optimize maximize minimize \\
    --reference-point 0.0 10.0 \\
    --n_gen 500 --pop_size 100 --output_dir results_mo_drug
        """
    )

    # Calculator options
    parser.add_argument('--calculator', type=str, default=None,
                        choices=['dft', 'cc', 'semiempirical', 'xtb'],
                        help='Calculator type (required for qc mode)')
    parser.add_argument('--basis', type=str, default="6-31G",
                        help='Basis set (for DFT/CC)')
    parser.add_argument('--functional', type=str, default="B3LYP",
                        help='Functional (for DFT/CC)')
    parser.add_argument('--method', type=str, default="full_tensor",
                        help='NLO calculation method')
    parser.add_argument('--se-method', type=str, default="PM7",
                        help='Semiempirical method (PM6, PM7, AM1, etc.)')
    parser.add_argument('--xtb-method', type=str, default="GFN2-xTB",
                        help='xTB method')
    parser.add_argument('--field-strength', type=float, default=0.001,
                        help='Field strength for finite field method')

    # Population / generation options
    parser.add_argument('--pop_size', type=int, default=100,
                        help='Number of molecules for archive initialisation')
    parser.add_argument('--n_gen', type=int, default=200,
                        help='Number of MO-CMA-MAE generations')
    parser.add_argument('--log_frequency', type=int, default=10,
                        help='How often to log progress (generations)')
    parser.add_argument('--save_frequency', type=int, default=50,
                        help='How often to save archive (generations)')
    parser.add_argument('--output_dir', type=str, default="mo_cma_mae_results",
                        help='Output directory for results')

    # Fitness mode
    parser.add_argument('--fitness-mode', type=str, default='qc',
                        choices=['qc', 'smartcadd', 'rdkit'],
                        help='Fitness evaluation mode: qc (xTB/DFT), smartcadd (docking/descriptors), rdkit (pure-RDKit drug oracle)')
    parser.add_argument('--smartcadd-path', type=str, default=None,
                        help='Path to SmartCADD repository')
    parser.add_argument('--smartcadd-mode', type=str, default='descriptors',
                        choices=['descriptors', 'docking'],
                        help='SmartCADD evaluation mode')
    parser.add_argument('--protein-code', type=str, default=None,
                        help='PDB code for docking target')
    parser.add_argument('--protein-path', type=str, default=None,
                        help='Local path to protein PDB file')
    parser.add_argument('--alert-collection', type=str, default=None,
                        help='Path to ADMET alert collection CSV')
    parser.add_argument('--atom-set', type=str, default=None,
                        choices=['nlo', 'drug'],
                        help='Atom set for mutation/validation')
    # Objectives — required unless --problem is used
    parser.add_argument('--objectives', type=str, nargs='+', default=None,
                        help='Objective keys (e.g. beta_gamma_ratio total_energy_atom_ratio). '
                             'Required unless --problem is specified.')
    parser.add_argument('--optimize', type=str, nargs='+', default=None,
                        help='Optimization direction per objective: maximize or minimize. '
                             'Required unless --problem is specified.')
    parser.add_argument('--reference-point', type=float, nargs='+', default=None,
                        help='Hypervolume reference point (one value per objective). '
                             'Required unless --problem is specified.')
    parser.add_argument('--problem', type=str, default=None,
                        help='Named problem preset (e.g. nlo_4obj, drug_2obj). '
                             'Fills default objectives, optimize directions, reference point, '
                             'and measure bounds. All explicit CLI flags override preset values.')

    # CVT archive options (CVT always used for MO-CMA-MAE)
    parser.add_argument('--n-centroids', type=int, default=100,
                        help='Number of CVT centroids (default: 100)')
    parser.add_argument('--cvt-samples', type=int, default=50000,
                        help='Random samples for CVT k-means seeding (default: 50000)')
    parser.add_argument('--cvt-measures', type=str, default='embedding',
                        choices=['structural', 'embedding', 'property'],
                        help='CVT measure type: structural (num_atoms/bonds), '
                             'embedding (ChemBERTa UMAP), or '
                             'property (fast oracle proxies, e.g. qed + sa_score)')
    parser.add_argument('--property-measure-keys', type=str, nargs='+', default=None,
                        help='Property keys to use as CVT behavioral dimensions when '
                             '--cvt-measures property (e.g. qed sa_score for drug; '
                             'homo_lumo_gap alpha_mean for NLO). '
                             'Keys must be present in evaluate_fn output.')
    parser.add_argument('--embedding-model', type=str,
                        default='DeepChem/ChemBERTa-77M-MTR',
                        help='HuggingFace model name for molecular embeddings')
    parser.add_argument('--embedding-dims', type=int, default=10,
                        help='UMAP output dimensionality for embedding measures (default: 10)')
    parser.add_argument('--embedding-device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda', 'mps'],
                        help='Device for embedding model (default: auto)')
    parser.add_argument('--embedding-sample-size', type=int, default=10000,
                        help='Molecules used to fit the UMAP embedder (default: 10000)')
    parser.add_argument('--measure-bounds', type=float, nargs='+', default=None,
                        help='Measure bounds as pairs: min1 max1 min2 max2. '
                             'For structural: atom/bond ranges. '
                             'For property: range of each --property-measure-keys value. '
                             'Common defaults: qed [0 1], sa_score [1 10], '
                             'homo_lumo_gap [0 30], alpha_mean [0 300].')

    # CMA-MAE options
    parser.add_argument('--learning-rate', type=float, default=0.01,
                        help='T_e bisection discount factor α (default: 0.01). '
                             'Controls how slowly the per-cell threshold front T_e '
                             'advances toward each observed solution. '
                             'Lower = more conservative T_e expansion = more exploration. '
                             'Does NOT control the pyribs adaptation archive threshold '
                             '(that uses MAP-Elites / learning_rate=1.0).')
    parser.add_argument('--threshold-min', type=float, default=None,
                        help='Floor for cell thresholds (default: auto, 0.0 when learning_rate < 1)')
    parser.add_argument('--cma-batch-size', type=int, default=36,
                        help='Solutions per CMA-ES ask() call per emitter (default: 36)')
    parser.add_argument('--n-emitters', type=int, default=5,
                        help='Number of independent CMA-ES emitters (default: 5)')
    parser.add_argument('--sigma0', type=float, default=0.5,
                        help='Initial CMA-ES step size (default: 0.5)')


    # Domain config file
    parser.add_argument('--config', type=str, default=None,
                        help='Path to a YAML domain config file that sets default values '
                             'for all arguments. Command-line arguments override config values. '
                             'Example: --config configs/nlo_sa_xtb.yaml')

    # Other
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--recalculate', type=str, default=None,
                        help='Recalculate metrics from existing results directory')

    # ── Load domain config (must happen after all add_argument calls) ──────
    _pre = argparse.ArgumentParser(add_help=False)
    _pre.add_argument('--config', type=str, default=None)
    _known, _ = _pre.parse_known_args()
    if _known.config:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'molev_utils'))
        from config_loader import inject_config_defaults
        inject_config_defaults(parser, _known.config)
    # ── End config loading ─────────────────────────────────────────────────

    args = parser.parse_args()

    # Resolve --problem preset before validation (fills None args from preset)
    from problem_config import resolve_from_args
    _problem = resolve_from_args(args)
    if _problem is not None:
        if args.objectives is None:
            args.objectives = _problem.objective_keys
        if args.optimize is None:
            args.optimize = _problem.optimize_strings
        if args.reference_point is None:
            args.reference_point = _problem.reference_point
        if args.measure_bounds is None:
            args.measure_bounds = _problem.measure_bounds_flat

    # Validate objective args (required unless --problem was used)
    if not args.objectives:
        parser.error("--objectives is required unless --problem is specified")
    if not args.optimize:
        parser.error("--optimize is required unless --problem is specified")
    if args.reference_point is None:
        parser.error("--reference-point is required unless --problem is specified")
    if len(args.optimize) != len(args.objectives):
        parser.error("--optimize must have the same number of values as --objectives")
    if len(args.reference_point) != len(args.objectives):
        parser.error("--reference-point must have the same number of values as --objectives")

    # Recalculation mode
    if args.recalculate:
        print(f"Recalculating results from {args.recalculate}...")
        optimize_objectives = [
            ('max' if d.lower() == 'maximize' else 'min', None)
            for d in args.optimize
        ]
        archive_config = {
            'objective_keys': args.objectives,
            'optimize_objectives': optimize_objectives,
            'reference_point': args.reference_point,
            'measure_keys': (['num_atoms', 'num_bonds'] if args.cvt_measures == 'structural'
                             else [f'emb_{i}' for i in range(args.embedding_dims)]),
            'measure_ranges': _parse_measure_bounds(args.measure_bounds),
            'dims': [10, 10],
        }
        op.MOCMAMaeOptimizer.recalculate_from_database(args.recalculate, archive_config)
        return

    # Seed everything
    import random
    import numpy as np
    from rdkit import Chem, rdBase

    random.seed(args.seed)
    np.random.seed(args.seed)
    rdBase.SeedRandomNumberGenerator(args.seed)

    # Atom set
    if args.atom_set:
        atom_set = args.atom_set
    elif args.fitness_mode in ('smartcadd', 'rdkit'):
        atom_set = 'drug'
    else:
        atom_set = 'nlo'

    # Evaluation interface
    if args.fitness_mode == 'rdkit':
        from drug.drug_oracle import DrugOracle
        eval_interface = DrugOracle()
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

    generator = MoleculeGenerator(seed=args.seed, atom_set=atom_set)

    # Objectives
    objective_keys = args.objectives
    optimize_objectives = [
        ('max' if d.lower() == 'maximize' else 'min', None)
        for d in args.optimize
    ]
    reference_point = np.array(args.reference_point)

    # ----------------------------------------------------------------
    # Embedder / measure setup
    # ----------------------------------------------------------------
    # The embedder is always initialised: it is used both as the CMA-ES search
    # space (via _decode_to_smiles nearest-neighbour) and, when
    # --cvt-measures embedding, as the CVT behavioural descriptor.
    cvt_seed_data = None
    from molecular_embedder import MolecularEmbedder
    print(f"Fitting ChemBERTa embedder on {args.embedding_sample_size} molecules...")
    _emb_raw = generator.generate_initial_population(args.embedding_sample_size)
    _emb_smiles = [generator.decode_to_smiles(s) for s in _emb_raw]
    _emb_smiles = [s for s in _emb_smiles if s is not None]
    embedder = MolecularEmbedder(
        model_name=args.embedding_model,
        n_components=args.embedding_dims,
        device=args.embedding_device,
        random_state=args.seed,
    )
    embedder.fit(_emb_smiles)
    print(f"Embedder fitted (n_components={args.embedding_dims}).")

    if args.cvt_measures == 'embedding':
        measure_keys = embedder.get_measure_keys()
        measure_bounds = embedder.get_measure_bounds()
        cvt_seed_data = embedder.get_fitted_embeddings()
        print(f"Embedding CVT measures: {measure_keys}")
    elif args.cvt_measures == 'property':
        # Property-informed behavioral descriptors:
        # cells in the archive correspond to chemically meaningful property niches
        # (e.g. HOMO-LUMO variation for NLO, drug-likeness landscape for drug design)
        # rather than structural variation.
        #
        # CVT seeding uses uniform random samples in property space — identical to the
        # structural path. No extra oracle calls are needed for seeding.
        # During optimisation the measure_keys are looked up directly from each
        # molecule's evaluate_fn output, so no new evaluation code is required.
        if not args.property_measure_keys:
            parser.error(
                "--property-measure-keys is required when --cvt-measures property. "
                "Example for drug: --property-measure-keys qed sa_score "
                "Example for NLO:  --property-measure-keys homo_lumo_gap alpha_mean"
            )
        measure_keys = args.property_measure_keys
        measure_bounds = _property_bounds_for_keys(measure_keys, args.measure_bounds)
        # cvt_seed_data stays None → uniform random seeding (same as structural)
        print(f"Property-informed CVT measures: {measure_keys}")
        print(f"  Bounds: {measure_bounds}")
    else:
        measure_keys = ['num_atoms', 'num_bonds']
        measure_bounds = _parse_measure_bounds(args.measure_bounds)

    # ----------------------------------------------------------------
    # evaluate_fn
    # ----------------------------------------------------------------
    def _error_props():
        base = {
            'smiles': None, 'num_atoms': 0.0, 'num_bonds': 0.0,
            'num_atoms_bin': 0, 'num_bonds_bin': 0,
            'error': 'Invalid molecule',
        }
        if embedder is not None:
            for k in embedder.get_measure_keys():
                base[k] = 0.0
        if args.cvt_measures == 'property' and args.property_measure_keys:
            for k in args.property_measure_keys:
                base.setdefault(k, 0.0)
        return base

    def evaluate_solution(smiles: str):
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
        return generator.decode_to_smiles(pop[0])

    # ----------------------------------------------------------------
    # threshold_min
    # ----------------------------------------------------------------
    # threshold_min is the floor for the HVI signal passed to scheduler.tell()
    # for failed decodes.  0.0 is the correct floor since valid HVI >= 0.
    threshold_min = args.threshold_min if args.threshold_min is not None else 0.0

    # ----------------------------------------------------------------
    # pyribs imports
    # ----------------------------------------------------------------
    try:
        from ribs.archives import CVTArchive
        from ribs.emitters import EvolutionStrategyEmitter
        from ribs.schedulers import Scheduler
    except ImportError:
        print("ERROR: pyribs is not installed. Run: pip install ribs")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # ----------------------------------------------------------------
    # CVT centroids (shared between adaptation archive and result archive)
    # ----------------------------------------------------------------
    from scipy.cluster.vq import kmeans2 as _kmeans2

    if cvt_seed_data is not None:
        print(f"Computing {args.n_centroids} CVT centroids from embedding data...")
        kmeans_input = cvt_seed_data
    else:
        # Generate uniform random samples so both archives share identical centroids.
        # (pyribs' internal k-means and scipy's kmeans2 differ — must pre-compute
        # and pass the same centroid array to both archives.)
        print(f"Computing {args.n_centroids} CVT centroids from uniform random samples...")
        rng = np.random.RandomState(args.seed)
        kmeans_input = np.column_stack([
            rng.uniform(lo, hi, size=args.cvt_samples)
            for lo, hi in measure_bounds
        ])

    centroids, _ = _kmeans2(kmeans_input, args.n_centroids,
                            minit='points', seed=args.seed)

    # ----------------------------------------------------------------
    # Adaptation archive (pyribs CVTArchive, MAP-Elites on HVI)
    # ----------------------------------------------------------------
    # learning_rate=1.0 → standard MAP-Elites: accepts z when HVI > best-ever per cell.
    # T_e (in the optimizer) is the sole annealing mechanism; the user's --learning-rate
    # (α) controls only the T_e bisection discount, not pyribs' internal threshold.
    # threshold_min=0.0 → require HVI > 0 to enter the adaptation archive.
    cma_archive = CVTArchive(
        solution_dim=args.embedding_dims,
        centroids=centroids,
        ranges=measure_bounds,
        learning_rate=1.0,
        threshold_min=0.0,
        seed=args.seed,
    )

    emitters = [
        EvolutionStrategyEmitter(
            archive=cma_archive,
            x0=np.zeros(args.embedding_dims),
            sigma0=args.sigma0,
            ranker='imp',
            selection_rule='mu',
            restart_rule='basic',
            batch_size=args.cma_batch_size,
            seed=args.seed + i,
        )
        for i in range(args.n_emitters)
    ]
    scheduler = Scheduler(archive=cma_archive, emitters=emitters)

    # ----------------------------------------------------------------
    # Result archive (CVTMOMEArchive, Pareto fronts)
    # ----------------------------------------------------------------
    from cvt_archive import CVTMOMEArchive

    result_archive = CVTMOMEArchive(
        n_centroids=args.n_centroids,
        measure_keys=measure_keys,
        objective_keys=objective_keys,
        measure_bounds=measure_bounds,
        optimize_objectives=optimize_objectives,
        cvt_samples=args.cvt_samples,
        random_state=args.seed,
        seed_data=cvt_seed_data,
        reference_point=reference_point,
        precomputed_centroids=centroids,   # guarantees same tessellation as cma_archive
    )

    # ----------------------------------------------------------------
    # Print configuration
    # ----------------------------------------------------------------
    print(f"\nMO-CMA-MAE configuration:")
    measures_label = args.cvt_measures
    if args.cvt_measures == 'property' and args.property_measure_keys:
        measures_label = f"property({'+'.join(args.property_measure_keys)})"
    print(f"  Archive type:  CVT ({measures_label} measures, {args.n_centroids} centroids)")
    print(f"  Embed dim:     {args.embedding_dims} (ChemBERTa UMAP)")
    print(f"  Measure keys:  {measure_keys}")
    print(f"  Measure bounds:{measure_bounds}")
    print(f"  Objectives:    {list(zip(objective_keys, [o[0] for o in optimize_objectives]))}")
    print(f"  Ref point:     {reference_point.tolist()}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Emitters:      {args.n_emitters} × batch {args.cma_batch_size} "
          f"= {args.n_emitters * args.cma_batch_size} evals/gen")
    # ----------------------------------------------------------------
    # Run
    # ----------------------------------------------------------------
    optimizer = op.MOCMAMaeOptimizer(
        scheduler=scheduler,
        result_archive=result_archive,
        embedder=embedder,
        mutate_fn=generator.mutate_as_smiles,
        generate_fn=generate_fn,
        evaluate_fn=evaluate_solution,
        measure_keys=measure_keys,
        objective_keys=objective_keys,
        optimize_objectives=optimize_objectives,
        reference_point=reference_point,
        output_dir=args.output_dir,
        random_init_size=args.pop_size,
        threshold_min=threshold_min,
        learning_rate=args.learning_rate,
        n_centroids=args.n_centroids,
    )

    optimizer.run(
        n_generations=args.n_gen,
        log_frequency=args.log_frequency,
        save_frequency=args.save_frequency,
    )

    best = optimizer.get_best_solutions(n=5)
    print(f"\nTop solutions from global Pareto front:")
    for entry in best:
        obj_str = ', '.join(f"{k}={entry['properties'].get(k, '?'):.3f}"
                            for k in objective_keys)
        print(f"  {entry['solution']}: {obj_str}")

    if args.fitness_mode == 'rdkit':
        import json
        pmo_stats = eval_interface.get_pmo_stats()
        pmo_path = os.path.join(args.output_dir, 'pmo_stats.json')
        with open(pmo_path, 'w') as _f:
            json.dump(pmo_stats, _f, indent=2)
        print(f"\nPMO statistics saved to {pmo_path}")
        print(f"  top1_auc={pmo_stats['top1_auc']:.4f}  top10_auc={pmo_stats['top10_auc']:.4f}"
              f"  top100_auc={pmo_stats['top100_auc']:.4f}")
        print(f"  final_top1={pmo_stats['final_top1']:.4f}  n_evaluations={pmo_stats['n_evaluations']}"
              f"  n_valid={pmo_stats['n_valid']}")


def _parse_measure_bounds(measure_bounds_flat):
    """Parse --measure-bounds into list of (min, max) tuples for structural measures."""
    if measure_bounds_flat:
        pairs = list(zip(measure_bounds_flat[::2], measure_bounds_flat[1::2]))
        return [(float(lo), float(hi)) for lo, hi in pairs]
    return [(1.0, 50.0), (0.0, 60.0)]  # default: num_atoms, num_bonds


from problem_config import bounds_for_keys as _property_bounds_for_keys  # noqa: E402


if __name__ == "__main__":
    main()
