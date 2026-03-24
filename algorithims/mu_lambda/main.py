"""
Main script for (μ+λ) Evolution Strategy molecular optimization
Supports both NLO (quantum chemistry) and drug design (SmartCADD) modes.
"""

import sys
import os
import argparse
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'molev_utils')))

from optimizer import MuLambdaOptimizer
from molecule_generator import MoleculeGenerator


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('mu_lambda.log'),
            logging.StreamHandler()
        ]
    )


def main():
    parser = argparse.ArgumentParser(
        description="(μ+λ) Evolution Strategy for Molecular Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Supported Modes:
  qc         - Quantum chemistry (NLO) optimization
  smartcadd  - Drug design optimization via SmartCADD

Supported Calculators (qc mode):
  dft          - DFT calculations (requires functional and basis)
  cc           - Coupled cluster calculations (requires functional and basis)
  semiempirical - Semiempirical methods (PM6, PM7, AM1, etc.)
  xtb          - xTB methods (GFN2-xTB, etc.)

Examples:
  # NLO: Maximize beta_gamma_ratio using DFT
  python main.py --fitness-mode qc --calculator dft --functional HF --basis 3-21G \\
                 --mu 20 --lambda 40 --n-gen 100 \\
                 --objective beta_gamma_ratio --maximize

  # Drug design: Maximize QED using SmartCADD descriptors
  python main.py --fitness-mode smartcadd --smartcadd-mode descriptors \\
                 --mu 20 --lambda 40 --n-gen 100 \\
                 --objective qed --maximize

  # Drug design: Minimize SA score with docking
  python main.py --fitness-mode smartcadd --smartcadd-mode docking \\
                 --protein-code 1AQ1 \\
                 --mu 20 --lambda 40 --n-gen 100 \\
                 --objective sa_score --minimize

  # Recalculate from existing database
  python main.py --recalculate mu_lambda_results/ --objective beta_mean
        """
    )

    # Evolution strategy parameters
    parser.add_argument('--mu', type=int, default=20,
                       help='Number of parents (μ)')
    parser.add_argument('--lambda', type=int, dest='lambda_', default=40,
                       help='Number of offspring per generation (λ)')
    parser.add_argument('--n-gen', type=int, default=100,
                       help='Number of generations')

    # Objective
    parser.add_argument('--objective', type=str, default='beta_mean',
                       help='Property to optimize (beta_mean, homo_lumo_gap, qed, sa_score, etc.)')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--maximize', action='store_true', default=True,
                      help='Maximize the objective (default)')
    group.add_argument('--minimize', action='store_true',
                      help='Minimize the objective')

    # Fitness mode options
    parser.add_argument('--fitness-mode', type=str, default='qc',
                       choices=['qc', 'smartcadd'],
                       help='Fitness evaluation mode: "qc" for quantum chemistry, '
                            '"smartcadd" for drug-design evaluation')

    # Calculator options (qc mode)
    parser.add_argument('--calculator', type=str, required=False,
                       choices=['dft', 'cc', 'semiempirical', 'xtb'],
                       help='Calculator type (required for qc mode)')
    parser.add_argument('--basis', type=str, default="6-31G",
                       help='Basis set (for DFT/CC)')
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

    # SmartCADD options (drug mode)
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

    # Output options
    parser.add_argument('--output-dir', type=str, default="mu_lambda_results",
                       help='Output directory for results')
    parser.add_argument('--save-frequency', type=int, default=10,
                       help='How often to save population (generations)')
    parser.add_argument('--log-frequency', type=int, default=1,
                       help='How often to log progress (generations)')

    # Other options
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--initial-seeds', type=str, nargs='+',
                       help='Initial SMILES strings to seed population')

    # Recalculation option
    parser.add_argument('--recalculate', type=str, default=None,
                       help='Recalculate metrics from existing all_molecules_database.json')

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Handle recalculation mode
    if args.recalculate:
        logger.info(f"Recalculating results from {args.recalculate}...")
        maximize = not args.minimize  # Use minimize flag if set, otherwise maximize
        MuLambdaOptimizer.recalculate_from_database(
            args.recalculate,
            objective_property=args.objective,
            maximize=maximize
        )
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
        logger.info(f"Using SmartCADD evaluation (mode={args.smartcadd_mode})")
    else:
        # Validate calculator is provided for QC mode
        if not args.calculator:
            parser.error("--calculator is required when using --fitness-mode qc")

        from quantum_chemistry_interface import QuantumChemistryInterface
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
        logger.info(f"Using quantum chemistry evaluation (calculator={args.calculator})")

    # Initialize components
    logger.info("Initializing molecule generator...")
    generator = MoleculeGenerator(seed=args.seed, atom_set=atom_set, encoding=args.encoding)

    # Determine maximize
    maximize = not args.minimize

    # Create optimizer
    logger.info("Creating (μ+λ) optimizer...")
    optimizer = MuLambdaOptimizer(
        mu=args.mu,
        lambda_=args.lambda_,
        n_gen=args.n_gen,
        objective_property=args.objective,
        maximize=maximize,
        generator=generator,
        eval_interface=eval_interface,
        output_dir=args.output_dir,
        initial_seeds=args.initial_seeds,
        save_frequency=args.save_frequency,
        log_frequency=args.log_frequency,
        seed=args.seed,
        encoding=args.encoding,
        crossover_rate=args.crossover_rate
    )

    # Run optimization
    logger.info("\n" + "="*60)
    logger.info(f"Starting optimization ({args.fitness_mode} mode)")
    logger.info(f"Objective: {args.objective} ({'maximize' if maximize else 'minimize'})")
    logger.info(f"Atom set: {atom_set}")
    logger.info("="*60 + "\n")

    final_population = optimizer.run()

    # Print final results
    logger.info("\n" + "="*60)
    logger.info("Optimization Complete")
    logger.info("="*60)

    if maximize:
        best_ind = max(final_population, key=lambda x: x.fitness)
    else:
        best_ind = min(final_population, key=lambda x: x.fitness)

    logger.info(f"\nBest solution:")
    logger.info(f"  SMILES: {best_ind.smiles}")
    logger.info(f"  {args.objective}: {best_ind.fitness:.6e}")
    logger.info(f"  Generation: {best_ind.generation}")
    logger.info(f"  Number of atoms: {best_ind.natoms}")

    logger.info(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
