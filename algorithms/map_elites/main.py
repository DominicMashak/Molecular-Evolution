import sys
import os
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'molev_utils')))

import archive as ar
import optimizer as op

from molecule_generator import MoleculeGenerator
from quantum_chemistry_interface import QuantumChemistryInterface

def main():
    parser = argparse.ArgumentParser(
        description="MAP-Elites Molecular Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Supported Calculators:
  dft          - DFT calculations (requires functional and basis)
  cc           - Coupled cluster calculations (requires functional and basis)
  semiempirical - Semiempirical methods (PM6, PM7, AM1, etc.)
  xtb          - xTB methods (GFN2-xTB, etc.)

Examples:
  # DFT with HF/STO-3G
  python main.py --calculator dft --functional HF --basis STO-3G \\
                 --pop_size 50 --n_gen 100

  # CC with HF/STO-3G
  python main.py --calculator cc --functional HF --basis STO-3G \\
                 --pop_size 50 --n_gen 100

  # Semiempirical with PM7
  python main.py --calculator semiempirical --se-method PM7 \\
                 --pop_size 50 --n_gen 100

  # xTB with GFN2-xTB
  python main.py --calculator xtb --xtb-method GFN2-xTB \\
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
    
    # Population/generation options
    parser.add_argument('--pop_size', type=int, default=50, 
                       help='Initial population size (random_init_size)')
    parser.add_argument('--n_gen', type=int, default=100, 
                       help='Number of generations')
    parser.add_argument('--iterations_per_gen', type=int, default=10, 
                       help='Iterations per generation')
    parser.add_argument('--log_frequency', type=int, default=20, 
                       help='How often to log progress (generations)')
    parser.add_argument('--save_frequency', type=int, default=50,
                       help='How often to save archive (generations)')
    parser.add_argument('--output_dir', type=str, default="map_elites_results",
                       help='Output directory for results')
    
    # Fitness mode options
    parser.add_argument('--fitness-mode', type=str, default='qc',
                       choices=['qc', 'smartcadd'],
                       help='Fitness evaluation mode: "qc" for quantum chemistry, '
                            '"smartcadd" for drug-design evaluation')
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
                       choices=['smiles', 'selfies', 'bigsmiles', 'slices'],
                       help='Molecular/crystal string encoding (smiles, selfies, bigsmiles, or slices)')
    parser.add_argument('--bigsmiles-n-samples', type=int, default=3,
                       help='Molecules sampled per BigSMILES genotype; best QED kept. '
                            'Only used with --encoding bigsmiles.')
    parser.add_argument('--bigsmiles-dp', type=int, default=None,
                       help='Fixed degree of polymerisation for BigSMILES sampling. '
                            'None = use distribution in the string. '
                            'Only used with --encoding bigsmiles.')
    parser.add_argument('--crystal-element-set', type=str, default='oxides',
                       choices=['oxides', 'semiconductor', 'semiconductors', 'halide_perovskite'],
                       help='Element set for crystal mutations (--encoding slices only)')
    parser.add_argument('--crystal-hardness', action='store_true',
                       help='Compute Vickers hardness via elastic constants (adds ~12 SCF runs). '
                            'Only used with --encoding slices.')
    parser.add_argument('--crystal-phonon-stability', action='store_true',
                       help='Check dynamical stability via ph.x at Γ (adds ~1 SCF equivalent). '
                            'Only used with --encoding slices.')
    parser.add_argument('--crystal-mobility', action='store_true',
                       help='Compute carrier mobility via deformation potential (+4 QE SCF runs). '
                            'Only used with --encoding slices.')
    parser.add_argument('--crystal-hull-distance', action='store_true',
                       help='Compute convex hull distance via CHGNet. '
                            'Only used with --encoding slices.')
    parser.add_argument('--crystal-tc', action='store_true',
                       help='Predict critical temperature via ALIGNN (InvDesFlow/JARVIS). '
                            'Replaces QE-based evaluation with fast ML (~1 s/structure). '
                            'Objective key: tc. Requires: pip install alignn jarvis-tools. '
                            'Only used with --encoding slices.')
    parser.add_argument('--tc-threshold', type=float, default=5.0,
                       help='(--crystal-tc only) Minimum Tc (K) for superconductor candidate screening (default: 5.0).')
    parser.add_argument('--crossover-rate', type=float, default=0.0,
                       help='Probability of crossover vs mutation per offspring (0.0 = off, default)')
    parser.add_argument('--objective-key', type=str, default=None,
                       help='Objective key for MAP-Elites archive (default: beta_gamma_ratio for qc, '
                            'qed for smartcadd descriptors, docking_score for smartcadd docking)')
    _obj_dir = parser.add_mutually_exclusive_group()
    _obj_dir.add_argument('--maximize', action='store_true',
                       help='Maximise the objective (default for all objectives except docking_score)')
    _obj_dir.add_argument('--minimize', action='store_true',
                       help='Minimise the objective (values negated internally so archive always maximises)')

    # Other options
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--reference-point', type=float, nargs='+', default=None,
                       help='Reference point for hypervolume calculation (list of floats)')
    parser.add_argument('--problem', type=str, default=None,
                       help='Named problem preset (e.g. nlo_1obj_beta, drug_1obj_qed). '
                            'Fills default measure bounds. All explicit CLI flags override preset values.')

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
                       help='CVT measure type: structural (num_atoms/bonds) or embedding (ChemBERTa for molecules, MatText for crystals)')

    # Embedding options (used with --cvt-measures embedding)
    parser.add_argument('--embedding-model', type=str, default='DeepChem/ChemBERTa-77M-MTR',
                       help='HuggingFace model for molecular embeddings')
    parser.add_argument('--embedding-dims', type=int, default=10,
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
    parser.add_argument('--recalculate', type=str, default=None,
                       help='Recalculate archive and metrics from existing all_molecules_database.json in the specified directory')

    args = parser.parse_args()

    # Resolve --problem preset (mainly used here for measure_bounds defaults)
    from problem_config import resolve_from_args
    _problem = resolve_from_args(args)
    if _problem is not None:
        if args.measure_bounds is None:
            args.measure_bounds = _problem.measure_bounds_flat
        if args.reference_point is None:
            args.reference_point = _problem.reference_point

    # Handle recalculation mode
    if args.recalculate:
        print(f"Recalculating results from {args.recalculate}...")
        archive_config = {
            'measure_dims': [10, 10],
            'measure_keys': ['num_atoms_bin', 'num_bonds_bin'],
            'objective_key': 'beta_gamma_ratio'
        }
        op.MAPElitesOptimizer.recalculate_from_database(args.recalculate, archive_config)
        return

    # Set random seeds for reproducibility
    import random
    import numpy as np
    from rdkit import Chem, rdBase

    random.seed(args.seed)
    np.random.seed(args.seed)
    rdBase.SeedRandomNumberGenerator(args.seed)

    # Determine atom set
    if args.atom_set:
        atom_set = args.atom_set
    elif args.fitness_mode == 'smartcadd':
        atom_set = 'drug'
    else:
        atom_set = 'nlo'

    # Setup evaluation interface (skipped for crystal SLICES encoding)
    eval_interface = None
    crystal_eval = None
    if args.encoding == 'slices':
        if getattr(args, 'crystal_tc', False):
            from crystal_tc_interface import CrystalTcInterface
            crystal_eval = CrystalTcInterface(verbose=args.verbose,
                                              tc_threshold=getattr(args, 'tc_threshold', 5.0))
        else:
            from crystal_qe_interface import CrystalQEInterface
            crystal_eval = CrystalQEInterface(verbose=args.verbose,
                                              calc_hardness=getattr(args, 'crystal_hardness', False),
                                              calc_phonon_stability=getattr(args, 'crystal_phonon_stability', False),
                                              calc_mobility=getattr(args, 'crystal_mobility', False),
                                              calc_hull_distance=getattr(args, 'crystal_hull_distance', False))
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
    generator = MoleculeGenerator(seed=args.seed, atom_set=atom_set, encoding=args.encoding,
                                  n_samples=getattr(args, 'bigsmiles_n_samples', 3),
                                  dp=getattr(args, 'bigsmiles_dp', None),
                                  element_set=getattr(args, 'crystal_element_set', 'oxides'))

    # Initialize molecular embedder if using embedding-based CVT measures
    embedder = None
    cvt_seed_data = None
    # Resolve embedding model: auto-switch to MatText for SLICES crystals
    _emb_model = args.embedding_model
    _emb_tok = None
    _emb_input_fmt = 'smiles'
    if args.encoding == 'slices' and _emb_model == 'DeepChem/ChemBERTa-77M-MTR':
        _emb_model = 'n0w0f/MatText-slices-2m'
        _emb_tok = 'bert-base-uncased'
        _emb_input_fmt = 'slices'
    if args.archive_type == 'cvt' and args.cvt_measures == 'embedding':
        from molecular_embedder import MolecularEmbedder
        embedding_sample_raw = generator.generate_initial_population(args.embedding_sample_size)
        if args.encoding == 'slices':
            embedding_sample = embedding_sample_raw  # SLICES strings fed directly to MatText
        else:
            embedding_sample = [generator.decode_to_smiles(s) for s in embedding_sample_raw]
            embedding_sample = [s for s in embedding_sample if s is not None]
        embedder = MolecularEmbedder(
            model_name=_emb_model,
            n_components=args.embedding_dims,
            device=args.embedding_device,
            random_state=args.seed,
            input_format=_emb_input_fmt,
            tokenizer_name=_emb_tok,
        )
        embedder.fit(embedding_sample)
        # Retrieve fitted embeddings for data-driven CVT centroid generation
        cvt_seed_data = embedder.get_fitted_embeddings()

    def generate_solution():
        """Generate one SMILES string."""
        population = generator.generate_initial_population(1)
        return population[0] if population else None

    def mutate_solution(parent):
        """Mutate one SMILES string."""
        mutated = generator.mutate_multiple(parent)
        return mutated if mutated is not None else parent

    def evaluate_solution(solution):
        """Evaluate molecule/crystal using configured interface."""
        from rdkit import Chem

        if solution is None:
            return {
                'beta_mean': 0.0, 'qed': 0.0, 'sa_score': 10.0,
                'num_atoms_bin': 0, 'num_bonds_bin': 0,
                'num_atoms': 0, 'num_bonds': 0,
                'error': 'Invalid solution'
            }

        # Crystal branch: SLICES encoding uses a parallel evaluation path
        if generator.encoding == 'slices':
            structure = generator.decode_to_structure(solution)
            if structure is None:
                from molev_utils.slices_ops import crystal_error_props
                props = crystal_error_props()
                props['n_sites_bin'] = 0
                props['n_species_bin'] = 0
                return props
            props = crystal_eval.calculate(solution, structure)
            n_sites = props.get('n_sites', 0)
            n_species = props.get('n_species', 0)
            # Bin measures for grid archive (n_sites: 1-20→0-9, n_species: 1-6→0-5)
            props['n_sites_bin'] = min(9, max(0, (n_sites - 1) * 10 // 20))
            props['n_species_bin'] = min(5, max(0, n_species - 1))
            if embedder is not None:
                emb = embedder.embed(solution)
                for i, val in enumerate(emb):
                    props[f'emb_{i}'] = float(val)
            if negate and objective_key in props and props[objective_key] is not None:
                props[objective_key] = -float(props[objective_key])
            return props

        # Decode to SMILES for RDKit / evaluation (no-op when encoding='smiles')
        smiles = generator.decode_to_smiles(solution)
        if smiles is None:
            return {
                'beta_mean': 0.0, 'qed': 0.0, 'sa_score': 10.0,
                'num_atoms_bin': 0, 'num_bonds_bin': 0,
                'num_atoms': 0, 'num_bonds': 0,
                'error': 'SELFIES decode failed'
            }

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {
                'beta_mean': 0.0, 'qed': 0.0, 'sa_score': 10.0,
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

        # Add embedding dimensions if embedder is active
        if embedder is not None:
            emb = embedder.embed(smiles)
            for i, val in enumerate(emb):
                props[f'emb_{i}'] = float(val)

        # Negate objective for minimization (archive always maximises internally)
        if negate and objective_key in props and props[objective_key] is not None:
            props[objective_key] = -float(props[objective_key])

        return props

    # Determine objective key and direction
    if args.objective_key:
        objective_key = args.objective_key
    elif args.encoding == 'slices':
        objective_key = 'formation_energy'
    elif args.fitness_mode == 'smartcadd' and args.smartcadd_mode == 'docking':
        objective_key = 'docking_score'
    elif args.fitness_mode == 'smartcadd':
        objective_key = 'qed'
    else:
        objective_key = 'beta_gamma_ratio'
    # docking_score always minimized; formation_energy minimized by default for crystals
    if objective_key == 'docking_score' and not args.maximize:
        negate = True
    elif args.encoding == 'slices' and objective_key == 'formation_energy' and not args.maximize:
        negate = True
    else:
        negate = args.minimize and not args.maximize

    # Set default reference point if not provided (legacy: was [0.0, 50, 15.0])
    if args.reference_point is None:
        args.reference_point = [0.0]

    # Create archive
    if args.archive_type == 'cvt':
        from cvt_archive import CVTMAPElitesArchive
        if args.cvt_measures == 'embedding' and embedder is not None:
            cvt_measure_keys = embedder.get_measure_keys()
            cvt_measure_bounds = embedder.get_measure_bounds()
        elif args.measure_bounds:
            _mk = ['n_sites', 'n_species'] if args.encoding == 'slices' else ['num_atoms', 'num_bonds']
            cvt_measure_keys = _mk
            cvt_measure_bounds = list(zip(args.measure_bounds[::2], args.measure_bounds[1::2]))
        elif args.encoding == 'slices':
            cvt_measure_keys = ['n_sites', 'n_species']
            cvt_measure_bounds = [(1, 20), (1, 6)]
        else:
            from problem_config import PROPERTY_BOUNDS as _PB
            cvt_measure_keys = ['num_atoms', 'num_bonds']
            cvt_measure_bounds = [_PB['num_atoms'], _PB['num_bonds']]
        archive = CVTMAPElitesArchive(
            n_centroids=args.n_centroids,
            measure_keys=cvt_measure_keys,
            objective_key=objective_key,
            measure_bounds=cvt_measure_bounds,
            cvt_samples=args.cvt_samples,
            random_state=args.seed,
            seed_data=cvt_seed_data,
        )
        print(f"Using CVT archive with {args.n_centroids} centroids, "
              f"measures={cvt_measure_keys}")
    elif args.encoding == 'slices':
        archive = ar.MAPElitesArchive(
            measure_dims=[10, 6],
            measure_keys=['n_sites_bin', 'n_species_bin'],
            objective_key=objective_key
        )
    else:
        archive = ar.MAPElitesArchive(
            measure_dims=[10, 10],
            measure_keys=['num_atoms_bin', 'num_bonds_bin'],
            objective_key=objective_key
        )
    
    crossover_fn = generator.crossover_in_encoding if args.crossover_rate > 0.0 else None
    optimizer = op.MAPElitesOptimizer(
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
    
    # Get best solution
    best = optimizer.get_best_solution()
    if best:
        print(f"\nBest solution: {best['solution']}")
        print(f"Properties: {best['properties']}")
    
    # Show some example solutions from different regions
    print("\nSample solutions from archive:")
    for entry in random.sample(optimizer.archive.get_all_solutions(),
                               min(5, len(optimizer.archive))):
        props = entry['properties']
        if generator.encoding == 'slices':
            summary = (f"eform={props.get('formation_energy', 0.0):.3f} eV/atom, "
                       f"n_sites={props.get('n_sites', 0)}, "
                       f"n_species={props.get('n_species', 0)}")
        else:
            summary = (f"beta_mean={props.get('beta_mean', 0.0):.3f}, "
                       f"num_atoms={props.get('num_atoms', 0)}")
        print(f"  {entry['indices']}: {entry['solution'][:60]} ({summary})")    
    
    #optimizer 
    
if __name__ == "__main__":
    main()