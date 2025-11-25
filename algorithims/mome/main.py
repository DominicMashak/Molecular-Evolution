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
    parser.add_argument('--calculator', type=str, required=True, 
                       choices=['dft', 'cc', 'semiempirical', 'xtb'],
                       help='Calculator type')
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
                       default=['beta_mean', 'homo_lumo_gap'],
                       help='List of objectives to optimize (e.g., beta_mean homo_lumo_gap)')
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
    
    # Other options
    parser.add_argument('--seed', type=int, default=42, 
                       help='Random seed')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--reference-point', type=float, nargs='+',
                       default=None,
                       help='Reference point for hypervolume calculation (one value per objective)')

    # Recalculation option
    parser.add_argument('--recalculate', type=str,
                       help='Recalculate archive, HV, MOQD from existing all_molecules_database.json in the specified results directory')

    args = parser.parse_args()

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
        'alpha_range_distance', 'homo_lumo_gap_range_distance'
    ]
    for obj in args.objectives:
        if obj not in valid_objectives:
            raise ValueError(f"Invalid objective '{obj}'. Must be one of: {valid_objectives}")
    
    if len(args.objectives) < 2:
        raise ValueError("MOME requires at least 2 objectives")
    
    # Set reference point
    if args.reference_point is None:
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
            'total_energy_atom_ratio': -100.0,  # Energy per atom is typically negative
            'alpha_range_distance': 500.0,  # Far from target range (worst case)
            'homo_lumo_gap_range_distance': 100.0  # Far from target range (worst case)
        }
        args.reference_point = [ref_defaults[obj] for obj in args.objectives]
    elif len(args.reference_point) != len(args.objectives):
        raise ValueError(f"Reference point must have {len(args.objectives)} values")
    
    print(f"\n{'='*70}")
    print(f"MOME: Multi-Objective MAP-Elites")
    print(f"{'='*70}")
    print(f"Objectives: {args.objectives}")
    print(f"Reference point: {args.reference_point}")
    print(f"Max Pareto front size: {args.max_front_size}")
    print(f"{'='*70}\n")
    
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
    
    # Initialize molecule generator with seed (uses equal mutation weights by default)
    generator = MoleculeGenerator(seed=args.seed)
    
    # Initialize quantum chemistry interface
    qc_interface = QuantumChemistryInterface(
        calculator_type=args.calculator,
        calculator_kwargs=calculator_kwargs,
        method=args.method,
        field_strength=args.field_strength,
        verbose=args.verbose
    )
    
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
        """Evaluate molecule: returns all properties including multiple objectives."""
        from rdkit import Chem, rdBase
        
        if solution is None:
            return {
                **{obj: 0.0 for obj in args.objectives},
                'num_atoms_bin': 0,
                'num_bonds_bin': 0,
                'num_atoms': 0,
                'num_bonds': 0,
                'error': 'Invalid solution'
            }
        
        mol = Chem.MolFromSmiles(solution)
        if mol is None:
            return {
                **{obj: 0.0 for obj in args.objectives},
                'num_atoms_bin': 0,
                'num_bonds_bin': 0,
                'num_atoms': 0,
                'num_bonds': 0,
                'error': 'Invalid SMILES'
            }
        
        num_atoms = mol.GetNumAtoms()
        num_bonds = mol.GetNumBonds()
        
        # Run quantum chemistry calculation
        qc_result = qc_interface.calculate(solution)
        
        # Extract all properties
        beta_mean = qc_result.get('beta_mean', 0.0) or 0.0
        homo_lumo_gap = qc_result.get('homo_lumo_gap', 0.0) or 0.0
        dipole_moment = qc_result.get('dipole_moment', 0.0) or 0.0
        alpha_mean = qc_result.get('alpha_mean', 0.0) or 0.0
        gamma = qc_result.get('gamma', 0.0) or 0.0
        total_energy = qc_result.get('total_energy', 0.0) or 0.0

        # Extract derived properties
        beta_gamma_ratio = qc_result.get('beta_gamma_ratio', 0.0) or 0.0
        total_energy_atom_ratio = qc_result.get('total_energy_atom_ratio', 0.0) or 0.0
        alpha_range_distance = qc_result.get('alpha_range_distance', 0.0) or 0.0
        homo_lumo_gap_range_distance = qc_result.get('homo_lumo_gap_range_distance', 0.0) or 0.0

        # Bin measures for MAP-Elites grid
        num_atoms_bin = min(19, max(0, (num_atoms - 5) // 2))
        num_bonds_bin = min(19, max(0, (num_bonds - 5) // 2))

        return {
            'beta_mean': beta_mean,
            'homo_lumo_gap': homo_lumo_gap,
            'dipole_moment': dipole_moment,
            'alpha_mean': alpha_mean,
            'gamma': gamma,
            'total_energy': total_energy,
            'beta_gamma_ratio': beta_gamma_ratio,
            'total_energy_atom_ratio': total_energy_atom_ratio,
            'alpha_range_distance': alpha_range_distance,
            'homo_lumo_gap_range_distance': homo_lumo_gap_range_distance,
            'num_atoms_bin': num_atoms_bin,
            'num_bonds_bin': num_bonds_bin,
            'num_atoms': num_atoms,
            'num_bonds': num_bonds,
            'error': qc_result.get('error')
        }
    
    # Create MOME archive with optimize_objectives (same as NSGA-II)
    # Use command-line --optimize argument or default to maximize all
    if args.optimize:
        if len(args.optimize) != len(args.objectives):
            raise ValueError(f"Number of optimization directions ({len(args.optimize)}) must match number of objectives ({len(args.objectives)})")
        optimize_objectives = [(opt, None) for opt in args.optimize]
    else:
        # All objectives are maximized by default
        optimize_objectives = [('maximize', None)] * len(args.objectives)

    archive = ma.MOMEArchive(
        measure_dims=[20, 20],
        measure_keys=['num_atoms_bin', 'num_bonds_bin'],
        objective_keys=args.objectives,
        max_front_size=args.max_front_size,
        optimize_objectives=optimize_objectives
    )
    
    # Create MOME optimizer
    optimizer = mo.MOMEOptimizer(
        archive=archive,
        generate_fn=generate_solution,
        mutate_fn=mutate_solution,
        evaluate_fn=evaluate_solution,
        random_init_size=args.pop_size,
        output_dir=args.output_dir,
        reference_point=args.reference_point
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