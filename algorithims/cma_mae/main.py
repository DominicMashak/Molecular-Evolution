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

  # Drug-like molecules, SmartCADD fitness
  python main.py --fitness-mode smartcadd --smartcadd-mode descriptors \\
                 --atom-set drug --objective qed --maximize \\
                 --latent-dim 64 --vae-train-size 5000 --vae-epochs 50 \\
                 --n-gen 200 --pop_size 100 --output_dir results_drug

  # Load a pre-trained VAE (skip training)
  python main.py --vae-path results_nlo/vae.pt --atom-set nlo \\
                 --objective beta_mean --maximize --n-gen 200 --output_dir results_nlo_2
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
    parser.add_argument('--encoding', type=str, default='smiles',
                        choices=['smiles', 'selfies'],
                        help='Encoding for MoleculeGenerator (VAE always uses SELFIES internally)')

    # Objective
    parser.add_argument('--objective', type=str, default=None,
                        help='Objective key to optimise (e.g. beta_mean, qed, sa_score)')
    parser.add_argument('--maximize', action='store_true',
                        help='Maximise the objective (default)')
    parser.add_argument('--minimize', action='store_true',
                        help='Minimise the objective (negates values internally)')

    # Archive measures
    parser.add_argument('--measure-keys', type=str, nargs='+',
                        default=['num_atoms', 'num_bonds'],
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

    # CMA-MAE options
    parser.add_argument('--learning-rate', type=float, default=0.01,
                        help='CMA-MAE threshold annealing rate α (default: 0.01). '
                             'Lower = slower threshold rise = more exploration.')
    parser.add_argument('--threshold-min', type=float, default=None,
                        help='Floor for cell thresholds (default: -inf)')
    parser.add_argument('--cma-batch-size', type=int, default=36,
                        help='Solutions per CMA-ES ask() call per emitter (default: 36)')
    parser.add_argument('--n-emitters', type=int, default=5,
                        help='Number of independent CMA-ES emitters (default: 5)')
    parser.add_argument('--sigma0', type=float, default=0.5,
                        help='Initial CMA-ES step size (default: 0.5)')

    # Other
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--recalculate', type=str, default=None,
                        help='Recalculate metrics from existing results directory')

    args = parser.parse_args()

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

    generator = MoleculeGenerator(seed=args.seed, atom_set=atom_set, encoding=args.encoding)

    # Determine objective key and sign
    if args.objective:
        objective_key = args.objective
    elif args.fitness_mode == 'smartcadd':
        objective_key = 'qed'
    else:
        objective_key = 'beta_gamma_ratio'

    negate = args.minimize and not args.maximize

    # Build evaluate_fn
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
        return props

    def _error_props():
        return {
            'smiles': None, 'num_atoms': 0.0, 'num_bonds': 0.0,
            'num_atoms_bin': 0, 'num_bonds_bin': 0,
            'beta_mean': 0.0, 'qed': 0.0, 'sa_score': 10.0,
            'error': 'Invalid molecule',
        }

    def generate_fn():
        pop = generator.generate_initial_population(1)
        if not pop:
            return None
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
        from ribs.emitters import EvolutionStrategyEmitter
        from ribs.schedulers import Scheduler
    except ImportError:
        print("ERROR: pyribs is not installed. Run: pip install ribs")
        sys.exit(1)

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

    emitters = [
        EvolutionStrategyEmitter(
            archive=cma_archive,
            x0=np.zeros(args.latent_dim),
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
    print(f"  Latent dim:    {args.latent_dim}")
    print(f"  Archive dims:  {archive_dims}  (total cells: {eval('*'.join(map(str, archive_dims)))})")
    print(f"  Measure keys:  {args.measure_keys}")
    print(f"  Measure bounds:{measure_bounds}")
    print(f"  Objective:     {objective_key} ({'min' if negate else 'max'})")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Emitters:      {args.n_emitters} × batch {args.cma_batch_size} "
          f"= {args.n_emitters * args.cma_batch_size} evals/gen")

    optimizer = op.CMAMaeOptimizer(
        scheduler=scheduler,
        result_archive=result_archive,
        vae=vae,
        generate_fn=generate_fn,
        evaluate_fn=evaluate_solution,
        measure_keys=args.measure_keys,
        objective_key=objective_key,
        output_dir=args.output_dir,
        random_init_size=args.pop_size,
        threshold_min=threshold_min,
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
    # Use defaults for known keys, fallback for unknown
    return [defaults.get(k, (0.0, 1.0)) for k in measure_keys]


if __name__ == "__main__":
    main()
