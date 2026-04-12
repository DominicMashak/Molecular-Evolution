import sys
import os
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'molev_utils')))

import optimizer as op

from molecule_generator import MoleculeGenerator
from quantum_chemistry_interface import QuantumChemistryInterface


def main():
    parser = argparse.ArgumentParser(
        description="CMA-MAE Molecular Optimisation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
<<<<<<< Updated upstream
CMA-MAE uses a Molecular VAE to provide a continuous latent space for CMA-ES.

Each generation evaluates (n_emitters * cma_batch_size) molecules:
  - CMA-ES generates latent vectors z in the VAE's latent space
  - The VAE decoder maps z -> SELFIES -> molecule
  - Each molecule is scored by the fitness evaluator
  - pyribs archives update CMA-ES covariance + threshold annealing

Examples:
  # NLO molecules, QC fitness (xTB)
  python main.py --calculator xtb --atom-set nlo --objective beta_mean --maximize \\
                 --latent-dim 64 --vae-train-size 5000 --vae-epochs 50 \\
                 --n-gen 200 --pop_size 100 --output_dir results_nlo
=======
CMA-MAE uses a frozen ChemBERTa-2 MTR UMAP embedding space for CMA-ES.

Each generation evaluates (n_emitters * cma_batch_size) molecules:
  - CMA-ES generates latent vectors z in the 10D ChemBERTa UMAP space
  - Nearest-neighbour lookup + SMILES mutation decodes z to a molecule
  - Each molecule is scored by the fitness evaluator
  - pyribs archives update CMA-ES covariance + threshold annealing

For multi-objective CMA-MAE see: algorithims/mo_cma_mae/

Examples:
  # NLO molecules, QC fitness (HF/3-21G), grid archive
  python main.py --calculator dft --functional HF --basis 3-21G \\
                 --atom-set nlo --objective beta_gamma_ratio --maximize \\
                 --n_gen 500 --pop_size 100 --output_dir results_nlo

  # NLO, CVT archive with ChemBERTa embedding measures
  python main.py --calculator dft --functional HF --basis 3-21G \\
                 --atom-set nlo --objective beta_gamma_ratio --maximize \\
                 --archive-type cvt --cvt-measures embedding --n-centroids 100 \\
                 --n_gen 500 --pop_size 100 --output_dir results_nlo_cvt
>>>>>>> Stashed changes

  # Drug-like molecules, SmartCADD fitness
  python main.py --fitness-mode smartcadd --smartcadd-mode descriptors \\
                 --atom-set drug --objective qed --maximize \\
<<<<<<< Updated upstream
                 --latent-dim 64 --vae-train-size 5000 --vae-epochs 50 \\
                 --n-gen 200 --pop_size 100 --output_dir results_drug

  # Load a pre-trained VAE (skip training)
  python main.py --vae-path results_nlo/vae.pt --atom-set nlo \\
                 --objective beta_mean --maximize --n-gen 200 --output_dir results_nlo_2
=======
                 --n_gen 500 --pop_size 100 --output_dir results_drug
>>>>>>> Stashed changes
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
                        help='Number of CMA-MAE generations')
    parser.add_argument('--log_frequency', type=int, default=10,
                        help='How often to log progress (generations)')
    parser.add_argument('--save_frequency', type=int, default=50,
                        help='How often to save archive (generations)')
    parser.add_argument('--output_dir', type=str, default="cma_mae_results",
                        help='Output directory for results')

    # Fitness mode
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
<<<<<<< Updated upstream
    parser.add_argument('--encoding', type=str, default='smiles',
                        choices=['smiles', 'selfies'],
                        help='Encoding for MoleculeGenerator (VAE always uses SELFIES internally)')

    # Objective
    parser.add_argument('--objective', type=str, default=None,
                        help='Objective key to optimise (e.g. beta_mean, qed, sa_score)')
=======
    # Objective
    parser.add_argument('--objective', type=str, default=None,
                        help='Objective key to optimise (e.g. beta_gamma_ratio, qed)')
>>>>>>> Stashed changes
    parser.add_argument('--maximize', action='store_true',
                        help='Maximise the objective (default)')
    parser.add_argument('--minimize', action='store_true',
                        help='Minimise the objective (negates values internally)')
<<<<<<< Updated upstream
=======
    parser.add_argument('--reference-point', type=float, nargs='+', default=None,
                        help='Reference point for QD-score / HV tracking. '
                             'Auto-set from --problem preset when omitted.')
>>>>>>> Stashed changes

    # Archive measures
    parser.add_argument('--measure-keys', type=str, nargs='+',
                        default=['num_atoms', 'num_bonds'],
<<<<<<< Updated upstream
                        help='Property keys used as archive measures (must be continuous floats)')
    parser.add_argument('--measure-bounds', type=float, nargs='+', default=None,
                        help='Measure bounds as pairs: min1 max1 min2 max2 ... '
                             '(default: 1 50 0 60 for num_atoms/num_bonds)')
    parser.add_argument('--archive-dims', type=int, nargs='+', default=[10, 10],
                        help='Grid archive dimensions per measure (default: 10 10)')

    # VAE options
    parser.add_argument('--latent-dim', type=int, default=64,
                        help='VAE latent space dimensionality (default: 64)')
    parser.add_argument('--vae-hidden-dim', type=int, default=256,
                        help='VAE GRU hidden dimension (default: 256)')
    parser.add_argument('--vae-embed-dim', type=int, default=64,
                        help='VAE token embedding dimension (default: 64)')
    parser.add_argument('--vae-path', type=str, default=None,
                        help='Path to a pre-trained VAE checkpoint (.pt). '
                             'If omitted, a new VAE is trained from scratch.')
    parser.add_argument('--vae-train-size', type=int, default=5000,
                        help='Number of molecules to train the VAE on (default: 5000)')
    parser.add_argument('--vae-epochs', type=int, default=50,
                        help='VAE training epochs (default: 50)')
    parser.add_argument('--vae-batch-size', type=int, default=256,
                        help='VAE training batch size (default: 256)')
    parser.add_argument('--vae-device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda', 'mps'],
                        help='Device for VAE training/inference (default: auto)')
=======
                        help='Property keys used as archive measures. '
                             'Overridden by --cvt-measures embedding.')
    parser.add_argument('--measure-bounds', type=float, nargs='+', default=None,
                        help='Measure bounds as pairs: min1 max1 min2 max2 ... '
                             '(default: 1 50 0 60 for num_atoms/num_bonds)')
    parser.add_argument('--problem', type=str, default=None,
                        help='Named problem preset (e.g. nlo_1obj_beta, drug_1obj_qed). '
                             'Fills default measure bounds. All explicit CLI flags override preset values.')
    parser.add_argument('--archive-dims', type=int, nargs='+', default=[10, 10],
                        help='Grid archive dimensions per measure (default: 10 10)')

    # CVT archive options
    parser.add_argument('--archive-type', type=str, default='grid',
                        choices=['grid', 'cvt'],
                        help='Archive tessellation type (default: grid)')
    parser.add_argument('--n-centroids', type=int, default=100,
                        help='Number of CVT centroids (default: 100)')
    parser.add_argument('--cvt-samples', type=int, default=50000,
                        help='Random samples for CVT k-means seeding (default: 50000)')
    parser.add_argument('--cvt-measures', type=str, default='structural',
                        choices=['structural', 'embedding'],
                        help='CVT measure type: structural (num_atoms/bonds) or '
                             'embedding (ChemBERTa UMAP)')
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
>>>>>>> Stashed changes

    # CMA-MAE options
    parser.add_argument('--learning-rate', type=float, default=0.01,
                        help='CMA-MAE threshold annealing rate α (default: 0.01). '
                             'Lower = slower threshold rise = more exploration.')
    parser.add_argument('--threshold-min', type=float, default=None,
<<<<<<< Updated upstream
                        help='Floor for cell thresholds (default: -inf)')
=======
                        help='Floor for cell thresholds (default: auto, 0.0 when learning_rate < 1)')
>>>>>>> Stashed changes
    parser.add_argument('--cma-batch-size', type=int, default=36,
                        help='Solutions per CMA-ES ask() call per emitter (default: 36)')
    parser.add_argument('--n-emitters', type=int, default=5,
                        help='Number of independent CMA-ES emitters (default: 5)')
    parser.add_argument('--sigma0', type=float, default=0.5,
                        help='Initial CMA-ES step size (default: 0.5)')

<<<<<<< Updated upstream
=======
    # Domain config file
    parser.add_argument('--config', type=str, default=None,
                        help='Path to a YAML domain config file that sets default values '
                             'for all arguments. Command-line arguments override config values. '
                             'Example: --config configs/nlo_sa_xtb.yaml')

>>>>>>> Stashed changes
    # Other
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--recalculate', type=str, default=None,
                        help='Recalculate metrics from existing results directory')

<<<<<<< Updated upstream
    args = parser.parse_args()

=======
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

    # Resolve --problem preset (mainly used here for measure_bounds defaults)
    from problem_config import resolve_from_args
    _problem = resolve_from_args(args)
    if _problem is not None:
        if args.measure_bounds is None:
            args.measure_bounds = _problem.measure_bounds_flat
        if args.reference_point is None:
            args.reference_point = _problem.reference_point

>>>>>>> Stashed changes
    # Recalculation mode
    if args.recalculate:
        print(f"Recalculating results from {args.recalculate}...")
        archive_config = {
            'measure_keys': args.measure_keys,
            'measure_ranges': _parse_measure_bounds(args.measure_bounds, args.measure_keys),
            'dims': args.archive_dims,
            'objective_key': args.objective or 'qed',
        }
        op.CMAMaeOptimizer.recalculate_from_database(args.recalculate, archive_config)
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
    elif args.fitness_mode == 'smartcadd':
        atom_set = 'drug'
    else:
        atom_set = 'nlo'

    # Evaluation interface
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

<<<<<<< Updated upstream
    generator = MoleculeGenerator(seed=args.seed, atom_set=atom_set, encoding=args.encoding)

    # Determine objective key and sign
    if args.objective:
        objective_key = args.objective
=======
    generator = MoleculeGenerator(seed=args.seed, atom_set=atom_set)

    # Objective
    if args.objective:
        objective_key = args.objective
    elif args.fitness_mode == 'smartcadd' and args.smartcadd_mode == 'docking':
        objective_key = 'docking_score'
>>>>>>> Stashed changes
    elif args.fitness_mode == 'smartcadd':
        objective_key = 'qed'
    else:
        objective_key = 'beta_gamma_ratio'
<<<<<<< Updated upstream

    negate = args.minimize and not args.maximize

    # Build evaluate_fn
=======
    # docking_score is always minimized (lower kcal/mol = tighter binding)
    if objective_key == 'docking_score' and not args.maximize:
        negate = True
    else:
        negate = args.minimize and not args.maximize

    # ----------------------------------------------------------------
    # Embedder setup (always initialised for the decode pool)
    # ----------------------------------------------------------------
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

    cvt_seed_data = None
    if args.archive_type == 'cvt' and args.cvt_measures == 'embedding':
        measure_keys = embedder.get_measure_keys()
        measure_bounds = embedder.get_measure_bounds()
        cvt_seed_data = embedder.get_fitted_embeddings()
        print(f"Embedding CVT measures: {measure_keys}")
    else:
        measure_keys = args.measure_keys
        measure_bounds = _parse_measure_bounds(args.measure_bounds, measure_keys)

    # ----------------------------------------------------------------
    # evaluate_fn
    # ----------------------------------------------------------------
    def _error_props():
        base = {
            'smiles': None, 'num_atoms': 0.0, 'num_bonds': 0.0,
            'num_atoms_bin': 0, 'num_bonds_bin': 0,
            'beta_mean': 0.0, 'qed': 0.0, 'sa_score': 10.0,
            'error': 'Invalid molecule',
        }
        if embedder is not None:
            for k in embedder.get_measure_keys():
                base[k] = 0.0
        return base

>>>>>>> Stashed changes
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
        if negate and objective_key in props and props[objective_key] is not None:
            props[objective_key] = -float(props[objective_key])
<<<<<<< Updated upstream
        return props

    def _error_props():
        return {
            'smiles': None, 'num_atoms': 0.0, 'num_bonds': 0.0,
            'num_atoms_bin': 0, 'num_bonds_bin': 0,
            'beta_mean': 0.0, 'qed': 0.0, 'sa_score': 10.0,
            'error': 'Invalid molecule',
        }

=======
        if embedder is not None:
            emb = embedder.embed(smiles)
            for i, val in enumerate(emb):
                props[f'emb_{i}'] = float(val)
        return props

>>>>>>> Stashed changes
    def generate_fn():
        pop = generator.generate_initial_population(1)
        if not pop:
            return None
<<<<<<< Updated upstream
        smi = generator.decode_to_smiles(pop[0])
        return smi

    # Parse / default measure bounds
    measure_bounds = _parse_measure_bounds(args.measure_bounds, args.measure_keys)

    # Archive dims must match number of measure keys
    archive_dims = args.archive_dims
    if len(archive_dims) < len(args.measure_keys):
        archive_dims = archive_dims + [10] * (len(args.measure_keys) - len(archive_dims))
    archive_dims = archive_dims[:len(args.measure_keys)]

    # pyribs requires a finite threshold_min when learning_rate < 1.0 (CMA-MAE mode)
    if args.threshold_min is not None:
        threshold_min = args.threshold_min
    elif args.learning_rate < 1.0:
        threshold_min = 0.0  # sensible floor: only accept solutions with positive objective
    else:
        threshold_min = -float('inf')

    # Build pyribs archive and emitters
    try:
        from ribs.archives import GridArchive
=======
        return generator.decode_to_smiles(pop[0])

    # ----------------------------------------------------------------
    # threshold_min
    # ----------------------------------------------------------------
    if args.threshold_min is not None:
        threshold_min = args.threshold_min
    elif args.learning_rate < 1.0:
        threshold_min = 0.0
    else:
        threshold_min = -float('inf')

    # ----------------------------------------------------------------
    # pyribs imports
    # ----------------------------------------------------------------
    try:
        from ribs.archives import GridArchive, CVTArchive
>>>>>>> Stashed changes
        from ribs.emitters import EvolutionStrategyEmitter
        from ribs.schedulers import Scheduler
    except ImportError:
        print("ERROR: pyribs is not installed. Run: pip install ribs")
        sys.exit(1)

<<<<<<< Updated upstream
    from molecular_vae import MolecularVAE
    import numpy as np

    # Load or train VAE
    os.makedirs(args.output_dir, exist_ok=True)
    if args.vae_path:
        print(f"Loading pre-trained VAE from {args.vae_path}")
        vae = MolecularVAE.load(args.vae_path, device=args.vae_device)
    else:
        print(f"Training VAE on {args.vae_train_size} molecules...")
        vae_train_raw = generator.generate_initial_population(args.vae_train_size)
        vae_train_smiles = [generator.decode_to_smiles(m) for m in vae_train_raw]
        vae_train_smiles = [s for s in vae_train_smiles if s is not None]
        vae = MolecularVAE(
            latent_dim=args.latent_dim,
            hidden_dim=args.vae_hidden_dim,
            embed_dim=args.vae_embed_dim,
            device=args.vae_device,
        )
        vae.fit(
            vae_train_smiles,
            epochs=args.vae_epochs,
            batch_size=args.vae_batch_size,
            verbose=True,
        )
        vae_save_path = os.path.join(args.output_dir, 'vae.pt')
        vae.save(vae_save_path)

    # CMA-MAE archive (threshold annealing)
    cma_archive = GridArchive(
        solution_dim=args.latent_dim,
        dims=archive_dims,
        ranges=measure_bounds,
        learning_rate=args.learning_rate,
        threshold_min=threshold_min,
        seed=args.seed,
    )
    # Result archive (best-ever per cell, no annealing)
    result_archive = GridArchive(
        solution_dim=args.latent_dim,
        dims=archive_dims,
        ranges=measure_bounds,
        learning_rate=1.0,
        seed=args.seed,
    )
=======
    os.makedirs(args.output_dir, exist_ok=True)

    # ----------------------------------------------------------------
    # Build archives
    # ----------------------------------------------------------------
    if args.archive_type == 'cvt':
        if cvt_seed_data is not None:
            from scipy.cluster.vq import kmeans2 as _kmeans2
            print(f"Computing {args.n_centroids} CVT centroids from embedding data...")
            centroids, _ = _kmeans2(cvt_seed_data, args.n_centroids,
                                    minit='points', seed=args.seed)
        else:
            centroids = args.n_centroids

        cma_archive = CVTArchive(
            solution_dim=args.embedding_dims,
            centroids=centroids,
            ranges=measure_bounds,
            learning_rate=args.learning_rate,
            threshold_min=threshold_min,
            seed=args.seed,
        )
        result_archive = CVTArchive(
            solution_dim=args.embedding_dims,
            centroids=centroids,
            ranges=measure_bounds,
            learning_rate=1.0,
            seed=args.seed,
        )
    else:
        archive_dims = args.archive_dims
        if len(archive_dims) < len(measure_keys):
            archive_dims = archive_dims + [10] * (len(measure_keys) - len(archive_dims))
        archive_dims = archive_dims[:len(measure_keys)]

        cma_archive = GridArchive(
            solution_dim=args.embedding_dims,
            dims=archive_dims,
            ranges=measure_bounds,
            learning_rate=args.learning_rate,
            threshold_min=threshold_min,
            seed=args.seed,
        )
        result_archive = GridArchive(
            solution_dim=args.embedding_dims,
            dims=archive_dims,
            ranges=measure_bounds,
            learning_rate=1.0,
            seed=args.seed,
        )
>>>>>>> Stashed changes

    emitters = [
        EvolutionStrategyEmitter(
            archive=cma_archive,
<<<<<<< Updated upstream
            x0=np.zeros(args.latent_dim),
=======
            x0=np.zeros(args.embedding_dims),
>>>>>>> Stashed changes
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

    print(f"\nCMA-MAE configuration:")
<<<<<<< Updated upstream
    print(f"  Latent dim:    {args.latent_dim}")
    print(f"  Archive dims:  {archive_dims}  (total cells: {eval('*'.join(map(str, archive_dims)))})")
    print(f"  Measure keys:  {args.measure_keys}")
=======
    print(f"  Archive type:  {args.archive_type.upper()}"
          + (f" ({args.cvt_measures} measures, {args.n_centroids} centroids)"
             if args.archive_type == 'cvt' else ""))
    print(f"  Embed dim:     {args.embedding_dims} (ChemBERTa UMAP)")
    print(f"  Measure keys:  {measure_keys}")
>>>>>>> Stashed changes
    print(f"  Measure bounds:{measure_bounds}")
    print(f"  Objective:     {objective_key} ({'min' if negate else 'max'})")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Emitters:      {args.n_emitters} × batch {args.cma_batch_size} "
          f"= {args.n_emitters * args.cma_batch_size} evals/gen")

<<<<<<< Updated upstream
    optimizer = op.CMAMaeOptimizer(
        scheduler=scheduler,
        result_archive=result_archive,
        vae=vae,
        generate_fn=generate_fn,
        evaluate_fn=evaluate_solution,
        measure_keys=args.measure_keys,
=======
    reference_point = args.reference_point if args.reference_point is not None else [0.0]
    optimizer = op.CMAMaeOptimizer(
        scheduler=scheduler,
        result_archive=result_archive,
        embedder=embedder,
        mutate_fn=generator.mutate_as_smiles,
        generate_fn=generate_fn,
        evaluate_fn=evaluate_solution,
        measure_keys=measure_keys,
>>>>>>> Stashed changes
        objective_key=objective_key,
        output_dir=args.output_dir,
        random_init_size=args.pop_size,
        threshold_min=threshold_min,
<<<<<<< Updated upstream
=======
        reference_point=reference_point,
>>>>>>> Stashed changes
    )

    optimizer.run(
        n_generations=args.n_gen,
        log_frequency=args.log_frequency,
        save_frequency=args.save_frequency,
    )

    best = optimizer.get_best_solution()
    if best:
        print(f"\nBest solution: {best['smiles']}")
        print(f"  {objective_key}: {best.get(objective_key)}")


def _parse_measure_bounds(measure_bounds_flat, measure_keys):
    """Parse --measure-bounds into list of (min, max) tuples."""
    defaults = {
        'num_atoms': (1.0, 50.0),
        'num_bonds': (0.0, 60.0),
    }
    if measure_bounds_flat:
        pairs = list(zip(measure_bounds_flat[::2], measure_bounds_flat[1::2]))
        return [(float(lo), float(hi)) for lo, hi in pairs]
<<<<<<< Updated upstream
    # Use defaults for known keys, fallback for unknown
=======
>>>>>>> Stashed changes
    return [defaults.get(k, (0.0, 1.0)) for k in measure_keys]


if __name__ == "__main__":
    main()
