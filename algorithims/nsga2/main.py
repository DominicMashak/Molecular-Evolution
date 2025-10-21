#!/usr/bin/env python3
"""
NSGA-II Main Entry Point
Supports multi-objective optimization with flexible objective configuration
"""

import argparse
import logging
import sys
import os

logging.basicConfig(level=logging.INFO)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'molev_utils')))

from molecule_generator import MoleculeGenerator
from optimizer import NSGA2Optimizer

def parse_optimize_arg(arg):
    """Parse optimize argument: 'maximize' -> ('max', None), 'target:5.0' -> ('target', 5.0)"""
    if arg.lower() in ['maximize', 'max']:
        return ('max', None)
    elif arg.lower() in ['minimize', 'min']:
        return ('min', None)
    elif arg.startswith('target:') or arg.startswith('t:'):
        try:
            _, value = arg.split(':', 1)
            return ('target', float(value))
        except ValueError:
            raise ValueError(f"Invalid target format: {arg}")
    else:
        raise ValueError(f"Unknown optimize type: {arg}")


def validate_objectives(objectives, optimize_objectives):
    """Validate that objectives and optimization directions are valid"""
    valid_objectives = {
        'beta', 'beta_mean', 'natoms', 'energy', 'dipole', 
        'homo_lumo_gap', 'alpha', 'gamma', 'transition_dipole', 
        'oscillator_strength'
    }
    
    for obj in objectives:
        if obj not in valid_objectives:
            logging.warning(f"Unknown objective '{obj}'. Valid objectives: {valid_objectives}")
    
    if len(objectives) != len(optimize_objectives):
        raise ValueError(
            f"Number of objectives ({len(objectives)}) must match "
            f"number of optimization directions ({len(optimize_objectives)})"
        )
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="NSGA-II Multi-Objective Molecular Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Supported Objectives:
  beta, beta_mean       - First hyperpolarizability (a.u.)
  natoms                - Number of atoms
  energy                - Total energy (a.u.)
  dipole                - Dipole moment (a.u.)
  homo_lumo_gap         - HOMO-LUMO gap (eV)
  alpha                 - Polarizability (a.u.)
  gamma                 - Second hyperpolarizability (a.u.)
  transition_dipole     - Transition dipole moment (a.u.)
  oscillator_strength   - Oscillator strength

Examples:
  # 2 objectives: maximize beta, minimize natoms
  python main.py --objectives beta natoms --optimize maximize minimize
  
  # 3 objectives: maximize beta, minimize natoms, maximize HOMO-LUMO gap
  python main.py --objectives beta natoms homo_lumo_gap \\
                 --optimize maximize minimize maximize
  
  # 4 objectives with DFT
  python main.py --calculator dft --functional B3LYP --basis 6-31G \\
                 --objectives beta natoms homo_lumo_gap energy \\
                 --optimize maximize minimize maximize minimize \\
                 --pop_size 20 --n_gen 100
  
  # 5 objectives with target natoms
  python main.py --objectives beta natoms homo_lumo_gap alpha gamma \\
                 --optimize maximize target:15 maximize maximize maximize
  
  # Fast optimization with semiempirical
  python main.py --calculator semiempirical --se-method PM7 \\
                 --objectives beta natoms homo_lumo_gap \\
                 --optimize maximize minimize maximize \\
                 --pop_size 30 --n_gen 200
  
  # Disable stagnation response
  python main.py --objectives beta natoms --optimize maximize minimize \\
                 --no-stagnation-response
        """
    )
    
    # Calculator options
    parser.add_argument('--calculator', type=str, required=True, 
                       choices=['dft', 'cc', 'semiempirical', 'xtb'],
                       help='Calculator type')
    parser.add_argument('--basis', type=str, default="6-31G", 
                       help='Basis set (for DFT/CC)')
    parser.add_argument('--functional', type=str, default="B3LYP", 
                       help='Functional (for DFT)')
    parser.add_argument('--method', type=str, default=None, 
                       help='Method (for DFT/CC)')
    parser.add_argument('--se-method', type=str, default="PM7", 
                       help='Semiempirical method (PM6, PM7, AM1, etc.)')
    parser.add_argument('--xtb-method', type=str, default="GFN2-xTB",
                       help='xTB method')
    parser.add_argument('--other', type=str, default=None, 
                       help='Other calculator options')
    
    # Population/generation options
    parser.add_argument('--pop_size', type=int, default=100, 
                       help='Population size')
    parser.add_argument('--n_gen', type=int, default=100, 
                       help='Number of generations')
    parser.add_argument('--n-parents', type=int, default=50, 
                       help='Number of parents to select per generation')
    parser.add_argument('--n-children', type=int, default=50, 
                       help='Number of children to generate per generation')
    
    # Mutation options
    parser.add_argument('--adaptive_mutation', action='store_true', 
                       help='Use adaptive mutation (deprecated, always on)')
    parser.add_argument('--mutation_weights', type=str, default=None, 
                       help='JSON file with mutation weights')
    
    # Output options
    parser.add_argument('--plot_every', type=int, default=10, 
                       help='Plot every n generations')
    parser.add_argument('--output_dir', type=str, default="nsga2_results", 
                       help='Output directory for results')
    
    # Objective options
    parser.add_argument('--objectives', nargs='+', required=True,
                       help='List of objectives (e.g., beta natoms homo_lumo_gap)')
    parser.add_argument('--optimize', nargs='+', required=True,
                       help='Optimization types (e.g., maximize minimize target:15.0)')
    
    # Stagnation response options
    parser.add_argument('--no-stagnation-response', action='store_true',
                       help='Disable stagnation-adaptive mutation')
    parser.add_argument('--stagnation-threshold', type=int, default=5,
                       help='Generations without improvement to trigger stagnation response')
    parser.add_argument('--stagnation-strategy', type=str, default='hybrid',
                       choices=['hybrid', 'weight', 'atoms'],
                       help='Stagnation response strategy')
    
    # Other options
    parser.add_argument('--debug-mopac', action='store_true', 
                       help='Enable debugging output for MOPAC')
    parser.add_argument('--seed', type=int, default=None, 
                       help='Random seed for reproducibility')
    parser.add_argument('--field-strength', type=float, default=0.001,
                       help='Field strength for full_tensor method')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--max-natoms', type=int, default=50,
                       help='Maximum number of atoms allowed')
    
    # Archive options
    parser.add_argument('--archive-bins', type=int, default=10,
                       help='Number of bins per dimension for archive')
    parser.add_argument('--archive-max-size', type=int, default=1000,
                       help='Maximum archive size')
    
    args = parser.parse_args()
    
    # Parse optimization objectives
    try:
        optimize_objectives = [parse_optimize_arg(opt) for opt in args.optimize]
    except ValueError as e:
        parser.error(str(e))
    
    # Validate objectives and optimization
    try:
        validate_objectives(args.objectives, optimize_objectives)
    except ValueError as e:
        parser.error(str(e))
    
    # Log configuration
    logging.info("="*60)
    logging.info("NSGA-II Multi-Objective Optimization")
    logging.info("="*60)
    logging.info(f"Calculator: {args.calculator}")
    logging.info(f"Objectives ({len(args.objectives)}): {', '.join(args.objectives)}")
    logging.info(f"Optimization: {', '.join([opt[0] for opt in optimize_objectives])}")
    logging.info(f"Population size: {args.pop_size}")
    logging.info(f"Generations: {args.n_gen}")
    logging.info(f"Output directory: {args.output_dir}")
    logging.info("="*60)
    
    # Load mutation weights
    mutation_weights = None
    if args.mutation_weights:
        import json
        try:
            with open(args.mutation_weights, 'r') as f:
                mutation_weights = json.load(f)
            logging.info(f"Loaded mutation weights from {args.mutation_weights}")
        except Exception as e:
            logging.warning(f"Failed to load mutation weights: {e}")
    
    # Setup calculator kwargs
    calculator_kwargs = {}
    if args.basis:
        calculator_kwargs['basis'] = args.basis
    if args.functional:
        calculator_kwargs['functional'] = args.functional
    if args.method:
        calculator_kwargs['method'] = args.method
    if args.se_method:
        calculator_kwargs['se_method'] = args.se_method
    if args.xtb_method:
        calculator_kwargs['xtb_method'] = args.xtb_method
    if args.debug_mopac:
        calculator_kwargs['debug_mopac'] = True
    
    # Create generator
    generator = MoleculeGenerator(mutation_weights=mutation_weights)
    
    # Create optimizer
    optimizer = NSGA2Optimizer(
        generator=generator,
        calculator_type=args.calculator,
        calculator_kwargs=calculator_kwargs,
        method=args.method or "full_tensor",
        pop_size=args.pop_size,
        n_gen=args.n_gen,
        plot_every=args.plot_every,
        adaptive_mutation=True,
        output_dir=args.output_dir,
        objectives=args.objectives,
        optimize_objectives=optimize_objectives,
        seed=args.seed,
        n_parents=args.n_parents,
        n_children=args.n_children,
        max_natoms=args.max_natoms,
        archive_bins=args.archive_bins,
        archive_max_size=args.archive_max_size,
        enable_stagnation_response=not args.no_stagnation_response,
        stagnation_threshold=args.stagnation_threshold,
        stagnation_strategy=args.stagnation_strategy,
        verbose=args.verbose,
        field_strength=args.field_strength
    )
    
    # Run optimization
    try:
        optimizer.run()
        logging.info("\n" + "="*60)
        logging.info("OPTIMIZATION COMPLETED SUCCESSFULLY!")
        logging.info("="*60)
        logging.info(f"Results saved to: {args.output_dir}")
        logging.info("\nGenerated files:")
        logging.info("  - pareto_front_molecules.json")
        logging.info("  - all_molecules_database.json")
        logging.info("  - performance_metrics.json")
        logging.info("  - final_convergence.pdf")
        logging.info("  - pareto_front_gen_*.png")
        
    except KeyboardInterrupt:
        logging.info("\n" + "="*60)
        logging.info("OPTIMIZATION INTERRUPTED BY USER")
        logging.info("="*60)
        logging.info("Saving partial results...")
        optimizer.save_results()
        logging.info(f"Partial results saved to: {args.output_dir}")
        sys.exit(1)
        
    except Exception as e:
        logging.error("\n" + "="*60)
        logging.error("OPTIMIZATION FAILED")
        logging.error("="*60)
        logging.error(f"Error: {e}")
        logging.exception("Traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()