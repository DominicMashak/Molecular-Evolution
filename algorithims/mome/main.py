import sys
import os
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'molev_utils')))

import archive as ma
import optimizer as mo
import performance as mp
# import plotting as mplot  # Commented out to avoid import error due to syntax issue in plotting.py

from molecule_generator import MoleculeGenerator
from quantum_chemistry_interface import QuantumChemistryInterface

from problem_config import bounds_for_keys


def _property_bounds_for_keys(keys, measure_bounds_flat):
    """Return (min, max) pairs for each property key."""
    return bounds_for_keys(keys, measure_bounds_flat)


def main():
    parser = argparse.ArgumentParser(
        description="MOME (Multi-Objective MAP-Elites) Molecular Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
MOME extends MAP-Elites to optimize for multiple conflicting objectives simultaneously
while maintaining diversity in the descriptor space. Each cell stores a Pareto front
of solutions instead of a single solution.

Supported Calculators:
  dft          - DFT calculations (requires functional and basis)
  cc           - Coupled cluster calculations (requires functional and basis)
  semiempirical - Semiempirical methods (PM6, PM7, AM1, etc.)
  xtb          - xTB methods (GFN2-xTB, etc.)

Examples:
  # MOME with DFT, optimizing beta_mean and homo_lumo_gap
  python mome_main.py --calculator dft --functional HF --basis STO-3G \\
                      --objectives beta_mean homo_lumo_gap \\
                      --pop_size 50 --n_gen 100

  # MOME with semiempirical
  python mome_main.py --calculator semiempirical --se-method PM7 \\
                      --objectives beta_mean dipole_moment \\
                      --pop_size 50 --n_gen 100
        """
    )
    
    # Calculator options
    parser.add_argument('--calculator', type=str, required=False,
                       choices=['dft', 'cc', 'semiempirical', 'xtb'],
                       help='Calculator type (required for qc mode)')
    parser.add_argument('--basis', type=str, default="6-31G", 
                       help='Basis set (for DFT)')
    parser.add_argument('--functional', type=str, default="B3LYP", 
                       help='Functional (for DFT)')
    parser.add_argument('--method', type=str, default="full_tensor", 
                       help='Calculation method (full_tensor, finite_field, cphf)')
    parser.add_argument('--se-method', type=str, default="PM7", 
                       help='Semiempirical method (PM6, PM7, AM1, etc.)')
    parser.add_argument('--xtb-method', type=str, default="GFN2-xTB",
                       help='xTB method')
    parser.add_argument('--field-strength', type=float, default=0.001,
                       help='Field strength for finite field method')
    
    # MOME-specific options
    parser.add_argument('--objectives', type=str, nargs='+',
                       default=None,
                       help='List of objectives to optimize (e.g., beta_mean homo_lumo_gap). '
                            'Defaults to [beta_mean, homo_lumo_gap] unless --problem is set.')
    parser.add_argument('--optimize', type=str, nargs='+',
                       default=None,
                       help='Optimization direction for each objective (maximize or minimize). If not specified, all objectives are maximized.')
    parser.add_argument('--max-front-size', type=int, default=50,
                       help='Maximum size of Pareto front per cell')
    
    # Population/generation options
    parser.add_argument('--pop_size', type=int, default=50, 
                       help='Initial population size')
    parser.add_argument('--n_gen', type=int, default=100, 
                       help='Number of generations')
    parser.add_argument('--iterations_per_gen', type=int, default=10, 
                       help='Iterations per generation')
    parser.add_argument('--log_frequency', type=int, default=20, 
                       help='How often to log progress (generations)')
    parser.add_argument('--save_frequency', type=int, default=50,
                       help='How often to save archive (generations)')
    parser.add_argument('--output_dir', type=str, default="mome_results",
                       help='Output directory for results')
    
    # Fitness mode options
    parser.add_argument('--fitness-mode', type=str, default='qc',
                       choices=['qc', 'smartcadd'],
                       help='Fitness evaluation mode')
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
    parser.add_argument('--encoding', type=str, default='smiles',
                       choices=['smiles', 'selfies'],
                       help='Molecular string encoding used internally (smiles or selfies)')
    parser.add_argument('--crossover-rate', type=float, default=0.0,
                       help='Probability of crossover vs mutation per offspring (0.0 = off, default)')

    # Other options
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--reference-point', type=float, nargs='+',
                       default=None,
                       help='Reference point for hypervolume calculation (one value per objective)')
    parser.add_argument('--problem', type=str, default=None,
                       help='Named problem preset (e.g. nlo_4obj, drug_2obj). '
                            'Fills default objectives, optimize directions, reference point, '
                            'and measure bounds. All explicit CLI flags override preset values. '
                            f'Available: see molev_utils/problem_config.py list_presets()')

    # Archive tessellation options
    parser.add_argument('--archive-type', type=str, default='grid',
                       choices=['grid', 'cvt'],
                       help='Archive tessellation: grid (fixed bins) or cvt (Centroidal Voronoi)')
    parser.add_argument('--n-centroids', type=int, default=100,
                       help='Number of CVT cells (--archive-type cvt only)')
    parser.add_argument('--cvt-samples', type=int, default=50000,
                       help='Random samples for CVT centroid generation')
    parser.add_argument('--measure-bounds', type=float, nargs='+', default=None,
                       help='Descriptor bounds as pairs: min1 max1 min2 max2 ...')
    parser.add_argument('--cvt-measures', type=str, default='structural',
                       choices=['structural', 'embedding', 'property'],
                       help='CVT measure type: structural (num_atoms/bonds), '
                            'embedding (ChemBERTa UMAP), or '
                            'property (fast oracle proxies, e.g. qed + sa_score)')
    parser.add_argument('--property-measure-keys', type=str, nargs='+', default=None,
                       help='Property keys to use as CVT behavioral dimensions when '
                            '--cvt-measures property (e.g. qed sa_score for drug; '
                            'homo_lumo_gap alpha_mean for NLO). '
                            'Keys must be present in evaluate_fn output.')

    # Embedding options (used with --cvt-measures embedding)
    parser.add_argument('--embedding-model', type=str, default='DeepChem/ChemBERTa-77M-MTR',
                       help='HuggingFace model for molecular embeddings')
    parser.add_argument('--embedding-dims', type=int, default=8,
                       help='Number of UMAP dimensions for embedding-based CVT measures')
    parser.add_argument('--embedding-device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda', 'mps'],
                       help='Device for transformer inference (auto detects best available)')
    parser.add_argument('--embedding-sample-size', type=int, default=1000,
                       help='Number of molecules for UMAP fitting (generates random molecules to learn the embedding manifold)')
    # Legacy alias for backward compatibility
    parser.add_argument('--pca-sample-size', type=int, dest='embedding_sample_size',
                       help='(DEPRECATED: use --embedding-sample-size) Number of molecules for UMAP fitting')

    # Archive tessellation options
    parser.add_argument('--archive-type', type=str, default='grid',
                       choices=['grid', 'cvt'],
                       help='Archive tessellation: grid (fixed bins) or cvt (Centroidal Voronoi)')
    parser.add_argument('--n-centroids', type=int, default=100,
                       help='Number of CVT cells (--archive-type cvt only)')
    parser.add_argument('--cvt-samples', type=int, default=50000,
                       help='Random samples for CVT centroid generation')
    parser.add_argument('--measure-bounds', type=float, nargs='+', default=None,
                       help='Descriptor bounds as pairs: min1 max1 min2 max2 ...')
    parser.add_argument('--cvt-measures', type=str, default='structural',
                       choices=['structural', 'embedding'],
                       help='CVT measure type: structural (num_atoms/bonds) or embedding (ChemBERTa)')

    # Embedding options (used with --cvt-measures embedding)
    parser.add_argument('--embedding-model', type=str, default='DeepChem/ChemBERTa-77M-MTR',
                       help='HuggingFace model for molecular embeddings')
    parser.add_argument('--embedding-dims', type=int, default=8,
                       help='Number of UMAP dimensions for embedding-based CVT measures')
    parser.add_argument('--embedding-device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda', 'mps'],
                       help='Device for transformer inference (auto detects best available)')
    parser.add_argument('--embedding-sample-size', type=int, default=1000,
                       help='Number of molecules for UMAP fitting (generates random molecules to learn the embedding manifold)')
    # Legacy alias for backward compatibility
    parser.add_argument('--pca-sample-size', type=int, dest='embedding_sample_size',
                       help='(DEPRECATED: use --embedding-sample-size) Number of molecules for UMAP fitting')

    # Recalculation option
    parser.add_argument('--recalculate', type=str,
                       help='Recalculate archive, HV, MOQD from existing all_molecules_database.json in the specified results directory')

    args = parser.parse_args()

    # Resolve --problem preset (fills defaults; explicit CLI flags already win because
    # resolve_from_args only overwrites when the arg is still None)
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
    # Legacy default when neither --problem nor --objectives provided
    if args.objectives is None:
        args.objectives = ['beta_mean', 'homo_lumo_gap']

    # Handle recalculation mode
    if args.recalculate:
        mo.MOMEOptimizer.recalculate_from_database(args.recalculate)
        return

    # Set random seeds for reproducibility
    import random
    import numpy as np
    from rdkit import Chem, rdBase

    random.seed(args.seed)
    np.random.seed(args.seed)
    rdBase.SeedRandomNumberGenerator(args.seed)
    
    # Validate objectives
    valid_objectives = [
        'beta_mean', 'homo_lumo_gap', 'dipole_moment',
        'alpha_mean', 'gamma', 'total_energy',
        # Derived objectives
        'beta_gamma_ratio', 'total_energy_atom_ratio',
        'alpha_range_distance', 'homo_lumo_gap_range_distance',
        # SmartCADD drug-design objectives
        'qed', 'sa_score', 'docking_score', 'lipinski_violations',
        'mol_weight', 'logp', 'tpsa', 'admet_pass',
    ]
    for obj in args.objectives:
        if obj not in valid_objectives:
            raise ValueError(f"Invalid objective '{obj}'. Must be one of: {valid_objectives}")
    
    if len(args.objectives) < 2:
        raise ValueError("MOME requires at least 2 objectives")
    
    # Set reference point
    if args.reference_point is None:
<<<<<<< Updated upstream
        # Default reference points (should be below minimum expected values)
        ref_defaults = {
            'beta_mean': 0.0,
            'homo_lumo_gap': 0.0,
            'dipole_moment': 0.0,
            'alpha_mean': 0.0,
            'gamma': 0.0,
            'total_energy': -1000.0,
            # Derived objectives
            'beta_gamma_ratio': 0.0,
            'total_energy_atom_ratio': -100.0,
            'alpha_range_distance': 500.0,
            'homo_lumo_gap_range_distance': 100.0,
            # SmartCADD drug-design objectives
            'qed': 0.0,
            'sa_score': 10.0,
            'docking_score': 0.0,
            'lipinski_violations': 4.0,
            'mol_weight': 0.0,
            'logp': -5.0,
            'tpsa': 0.0,
            'admet_pass': 0.0,
        }
        args.reference_point = [ref_defaults.get(obj, 0.0) for obj in args.objectives]
=======
        # Auto-derive from PROPERTY_BOUNDS: worst case for each direction
        #   maximize → lower bound of physical range
        #   minimize → upper bound of physical range
        from problem_config import PROPERTY_BOUNDS as _PB
        optimize_map = dict(zip(args.objectives, args.optimize)) if args.optimize else {}
        refs = []
        for obj in args.objectives:
            direction = optimize_map.get(obj, 'maximize')
            lo, hi = _PB.get(obj, (0.0, 1.0))
            refs.append(lo if direction == 'maximize' else hi)
        args.reference_point = refs
>>>>>>> Stashed changes
    elif len(args.reference_point) != len(args.objectives):
        raise ValueError(f"Reference point must have {len(args.objectives)} values")
    
    print(f"\n{'='*70}")
    print(f"MOME: Multi-Objective MAP-Elites")
    print(f"{'='*70}")
    print(f"Objectives: {args.objectives}")
    print(f"Reference point: {args.reference_point}")
    print(f"Max Pareto front size: {args.max_front_size}")
    print(f"Archive type: {args.archive_type}")
    if args.archive_type == 'cvt':
        print(f"CVT centroids: {args.n_centroids}")
        print(f"CVT measures: {args.cvt_measures}")
        if args.cvt_measures == 'embedding':
            print(f"Embedding model: {args.embedding_model}")
            print(f"Embedding dims: {args.embedding_dims}")
    print(f"{'='*70}\n")
    
    # Determine atom set
    if args.atom_set:
        atom_set = args.atom_set
    elif args.fitness_mode == 'smartcadd':
        atom_set = 'drug'
    else:
        atom_set = 'nlo'

    # Setup evaluation interface
    if args.fitness_mode == 'smartcadd':
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
        # Setup calculator kwargs
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
            verbose=args.verbose
        )

    # Initialize molecule generator with seed
    generator = MoleculeGenerator(seed=args.seed, atom_set=atom_set, encoding=args.encoding)

    # Initialize molecular embedder if using embedding-based CVT measures
    embedder = None
    cvt_seed_data = None
<<<<<<< Updated upstream
=======
    property_measure_keys = None  # set when --cvt-measures property

    if args.archive_type == 'cvt' and args.cvt_measures == 'property':
        if not args.property_measure_keys:
            parser.error(
                "--property-measure-keys is required when --cvt-measures property. "
                "Example for drug: --property-measure-keys qed sa_score "
                "Example for NLO:  --property-measure-keys homo_lumo_gap alpha_mean"
            )
        property_measure_keys = args.property_measure_keys
        # cvt_seed_data stays None → uniform random seeding in property space

>>>>>>> Stashed changes
    if args.archive_type == 'cvt' and args.cvt_measures == 'embedding':
        from molecular_embedder import MolecularEmbedder
        # Generate random molecules to fit UMAP manifold (always decode to SMILES for embedder)
        embedding_sample_raw = generator.generate_initial_population(args.embedding_sample_size)
        embedding_sample_smiles = [generator.decode_to_smiles(s) for s in embedding_sample_raw]
        embedding_sample_smiles = [s for s in embedding_sample_smiles if s is not None]
        embedder = MolecularEmbedder(
            model_name=args.embedding_model,
            n_components=args.embedding_dims,
            device=args.embedding_device,
            random_state=args.seed,
        )
        embedder.fit(embedding_sample_smiles)  #fit() is the actual method, fit_pca is legacy alias
        # Retrieve the fitted embeddings for data-driven CVT centroid generation.
        # This ensures centroids are placed where molecules actually live in the
        # UMAP manifold rather than in uniformly sampled empty regions.
        cvt_seed_data = embedder.get_fitted_embeddings()

    # Pre-generate and save initial population for reproducibility tracking
    initial_population = generator.generate_initial_population(
        args.pop_size,
        save_to_file=True,
        seed_number=args.seed,
        algorithm_name="mome"
    )
    population_iter = iter(initial_population)
    initial_exhausted = False

    def generate_solution():
        """Generate one SMILES string."""
        nonlocal initial_exhausted
        if not initial_exhausted:
            try:
                return next(population_iter)
            except StopIteration:
                initial_exhausted = True
        # Fallback to generating new molecules
        population = generator.generate_initial_population(1)
        return population[0] if population else None

    def mutate_solution(parent):
        """Mutate one SMILES string."""
        mutated = generator.mutate_multiple(parent)
        return mutated if mutated is not None else parent

    def evaluate_solution(solution):
        """Evaluate molecule using configured interface (QC or SmartCADD)."""
        from rdkit import Chem

<<<<<<< Updated upstream
        if solution is None:
            return {
                **{obj: 0.0 for obj in args.objectives},
=======
        _prop_zeros = {k: 0.0 for k in (property_measure_keys or [])}

        if solution is None:
            return {
                **{obj: 0.0 for obj in args.objectives},
                **_prop_zeros,
>>>>>>> Stashed changes
                'num_atoms_bin': 0, 'num_bonds_bin': 0,
                'num_atoms': 0, 'num_bonds': 0,
                'error': 'Invalid solution'
            }

        # Decode to SMILES for RDKit / evaluation (no-op when encoding='smiles')
        smiles = generator.decode_to_smiles(solution)
        if smiles is None:
            return {
                **{obj: 0.0 for obj in args.objectives},
<<<<<<< Updated upstream
=======
                **_prop_zeros,
>>>>>>> Stashed changes
                'num_atoms_bin': 0, 'num_bonds_bin': 0,
                'num_atoms': 0, 'num_bonds': 0,
                'error': 'SELFIES decode failed'
            }

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {
                **{obj: 0.0 for obj in args.objectives},
<<<<<<< Updated upstream
=======
                **_prop_zeros,
>>>>>>> Stashed changes
                'num_atoms_bin': 0, 'num_bonds_bin': 0,
                'num_atoms': 0, 'num_bonds': 0,
                'error': 'Invalid SMILES'
            }

        num_atoms = mol.GetNumAtoms()
        num_bonds = mol.GetNumBonds()

        # Run evaluation
        result = eval_interface.calculate(smiles)

        # Bin measures for MAP-Elites grid (0-9)
        num_atoms_bin = min(9, max(0, (num_atoms - 5) // 3))
        num_bonds_bin = min(9, max(0, (num_bonds - 5) // 3))

        props = {
            'smiles': smiles,
            'num_atoms_bin': num_atoms_bin,
            'num_bonds_bin': num_bonds_bin,
            'num_atoms': num_atoms,
            'num_bonds': num_bonds,
            'error': result.get('error'),
        }
        if args.encoding == 'selfies':
            props['selfies'] = solution
        # Copy all properties from evaluation result
        for key, val in result.items():
            if key not in props and key != 'smiles':
                props[key] = val if val is not None else 0.0

        # If QC failed and objective keys are missing, assign reference point
        # values (worst acceptable fitness) so failed molecules don't dominate
        missing_objectives = [obj for obj in args.objectives if obj not in props]
        if missing_objectives:
            print(f"  WARNING: QC failed for {smiles}, missing: {missing_objectives}. Assigning reference point values.")
            for i, obj in enumerate(args.objectives):
                if obj not in props:
                    props[obj] = args.reference_point[i]

        # Add ChemBERTa embedding dimensions if embedder is active
        if embedder is not None:
            emb = embedder.embed(smiles)
            for i, val in enumerate(emb):
                props[f'emb_{i}'] = float(val)

        return props
    
    # Create MOME archive with optimize_objectives (same as NSGA-II)
    # Use command-line --optimize argument or default to maximize all
    if args.optimize:
        if len(args.optimize) != len(args.objectives):
            raise ValueError(f"Number of optimization directions ({len(args.optimize)}) must match number of objectives ({len(args.objectives)})")
        optimize_objectives = [(opt, None) for opt in args.optimize]
    else:
        # All objectives are maximized by default
        optimize_objectives = [('maximize', None)] * len(args.objectives)

    if args.archive_type == 'cvt':
        from cvt_archive import CVTMOMEArchive
        if args.cvt_measures == 'embedding' and embedder is not None:
            cvt_measure_keys = embedder.get_measure_keys()
            cvt_measure_bounds = embedder.get_measure_bounds()
<<<<<<< Updated upstream
=======
        elif args.cvt_measures == 'property' and property_measure_keys:
            # Property-informed behavioral space: archive cells correspond to
            # chemically meaningful property niches (e.g. QED × SA landscape).
            # CVT seeding uses uniform random samples in property space — no
            # extra oracle calls needed.
            cvt_measure_keys = property_measure_keys
            cvt_measure_bounds = _property_bounds_for_keys(
                property_measure_keys, args.measure_bounds
            )
            print(f"Property-informed CVT measures: {cvt_measure_keys}")
            print(f"  Bounds: {cvt_measure_bounds}")
>>>>>>> Stashed changes
        elif args.measure_bounds:
            cvt_measure_keys = ['num_atoms', 'num_bonds']
            cvt_measure_bounds = list(zip(args.measure_bounds[::2], args.measure_bounds[1::2]))
        else:
            cvt_measure_keys = ['num_atoms', 'num_bonds']
            cvt_measure_bounds = [(5, 35), (4, 40)]
        archive = CVTMOMEArchive(
            n_centroids=args.n_centroids,
            measure_keys=cvt_measure_keys,
            objective_keys=args.objectives,
            measure_bounds=cvt_measure_bounds,
            max_front_size=args.max_front_size,
            optimize_objectives=optimize_objectives,
            cvt_samples=args.cvt_samples,
            random_state=args.seed,
            seed_data=cvt_seed_data,
        )
    else:
        archive = ma.MOMEArchive(
            measure_dims=[10, 10],  # 10x10 = 100 cells, bin size 3 for 5-32 atom coverage
            measure_keys=['num_atoms_bin', 'num_bonds_bin'],
            objective_keys=args.objectives,
            max_front_size=args.max_front_size,
            optimize_objectives=optimize_objectives
        )
    
    # Create MOME optimizer
    crossover_fn = generator.crossover_in_encoding if args.crossover_rate > 0.0 else None
    optimizer = mo.MOMEOptimizer(
        archive=archive,
        generate_fn=generate_solution,
        mutate_fn=mutate_solution,
        evaluate_fn=evaluate_solution,
        random_init_size=args.pop_size,
        output_dir=args.output_dir,
        reference_point=args.reference_point,
        crossover_rate=args.crossover_rate,
        crossover_fn=crossover_fn
    )
    
    # Run optimization
    history = optimizer.run(
        n_generations=args.n_gen,
        iterations_per_generation=args.iterations_per_gen,
        log_frequency=args.log_frequency,
        save_frequency=args.save_frequency
    )
    
    # Print final statistics
    print(f"\n{'='*70}")
    print("Final Statistics")
    print(f"{'='*70}")
    stats = optimizer.get_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key:.<30} {value:.4f}")
        else:
            print(f"{key:.<30} {value}")
    print(f"{'='*70}\n")

    # Show best solutions from global Pareto front
    print("\nTop solutions from Global Pareto Front:")
    best_solutions = optimizer.get_best_solutions(n=5)
    for i, entry in enumerate(best_solutions, 1):
        print(f"\n{i}. {entry['solution']}")
        for j, obj in enumerate(args.objectives):
            print(f"   {obj}: {entry['objectives'][j]:.4f}")
        print(f"   Measures: {entry['indices']}")
    
    # # Generate additional plots  # Commented out to avoid using broken plotting module
    # print("\nGenerating additional plots...")
    # plotter = mplot.MOMEPlotter(optimizer.output_dir)
    # plotter.plot_global_pareto_front(archive)
    # plotter.plot_pareto_fronts_grid(archive, n_samples=9)
    # plotter.plot_objective_histograms(archive)
    
    print(f"\nAll results saved to: {optimizer.output_dir}")
    print("MOME optimization complete!")
    
if __name__ == "__main__":
    main()