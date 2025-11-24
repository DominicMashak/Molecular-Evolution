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
    
    # Other options
    parser.add_argument('--seed', type=int, default=42, 
                       help='Random seed')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--reference-point', type=float, nargs='+', default=[0.0, 50, 15.0],
                       help='Reference point for hypervolume calculation (list of floats)')

    # Recalculation option
    parser.add_argument('--recalculate', type=str, default=None,
                       help='Recalculate archive and metrics from existing all_molecules_database.json in the specified directory')

    args = parser.parse_args()

    # Handle recalculation mode
    if args.recalculate:
        print(f"Recalculating results from {args.recalculate}...")
        archive_config = {
            'measure_dims': [20, 20],
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
    
    def generate_solution():
        """Generate one SMILES string."""
        population = generator.generate_initial_population(1)
        return population[0] if population else None
    
    def mutate_solution(parent):
        """Mutate one SMILES string."""
        mutated = generator.mutate_multiple(parent)
        return mutated if mutated is not None else parent  # Fallback to parent if mutation fails
    
    def evaluate_solution(solution):
        """Evaluate molecule: fitness is beta_mean, measures are binned num_atoms and num_bonds."""
        from rdkit import Chem
        
        if solution is None:
            return {
                'beta_mean': 0.0,
                'num_atoms_bin': 0,
                'num_bonds_bin': 0,
                'num_atoms': 0,
                'num_bonds': 0,
                'homo_lumo_gap': 0.0,
                'error': 'Invalid solution'
            }
        
        mol = Chem.MolFromSmiles(solution)
        if mol is None:
            return {
                'beta_mean': 0.0,
                'num_atoms_bin': 0,
                'num_bonds_bin': 0,
                'num_atoms': 0,
                'num_bonds': 0,
                'homo_lumo_gap': 0.0,
                'error': 'Invalid SMILES'
            }
        
        num_atoms = mol.GetNumAtoms()
        num_bonds = mol.GetNumBonds()
        
        # Run quantum chemistry calculation
        qc_result = qc_interface.calculate(solution)
        
        beta_mean = qc_result.get('beta_mean', 0.0) or 0.0
        homo_lumo_gap = qc_result.get('homo_lumo_gap', 0.0) or 0.0
        
        # Bin measures for MAP-Elites (0-19) - finer granularity
        # Bin size 2 gives better coverage across typical molecule sizes (5-40 atoms)
        num_atoms_bin = min(19, max(0, (num_atoms - 5) // 2))  # Starts at 5 atoms, bin size 2
        num_bonds_bin = min(19, max(0, (num_bonds - 5) // 2))  # Starts at 5 bonds, bin size 2
        
        return {
            'beta_mean': beta_mean,
            'num_atoms_bin': num_atoms_bin,
            'num_bonds_bin': num_bonds_bin,
            'num_atoms': num_atoms,
            'num_bonds': num_bonds,
            'homo_lumo_gap': homo_lumo_gap,
            'dipole_moment': qc_result.get('dipole_moment', 0.0) or 0.0,
            'total_energy': qc_result.get('total_energy', 0.0) or 0.0,
            'alpha_mean': qc_result.get('alpha_mean', 0.0) or 0.0,
            'gamma': qc_result.get('gamma', 0.0) or 0.0,
            'beta_gamma_ratio': qc_result.get('beta_gamma_ratio', 0.0) or 0.0,
            'total_energy_atom_ratio': qc_result.get('total_energy_atom_ratio', 0.0) or 0.0,
            'alpha_range_distance': qc_result.get('alpha_range_distance', 0.0) or 0.0,
            'homo_lumo_gap_range_distance': qc_result.get('homo_lumo_gap_range_distance', 0.0) or 0.0,
            'error': qc_result.get('error')
        }
    
    # Create archive and optimizer
    archive = ar.MAPElitesArchive(
        measure_dims=[20, 20],  # Changed from [10, 10] to match new finer binning
        measure_keys=['num_atoms_bin', 'num_bonds_bin'],
        objective_key='beta_gamma_ratio'
    )
    
    optimizer = op.MAPElitesOptimizer(
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
    
    # Get best solution
    best = optimizer.get_best_solution()
    if best:
        print(f"\nBest solution: {best['solution']}")
        print(f"Properties: {best['properties']}")
    
    # Show some example solutions from different regions
    print("\nSample solutions from archive:")
    for entry in random.sample(optimizer.archive.get_all_solutions(), 
                               min(5, len(optimizer.archive))):
        print(f"  {entry['indices']}: {entry['solution']} "
              f"(beta_mean={entry['properties']['beta_mean']:.3f}, "
              f"num_atoms={entry['properties']['num_atoms']})")    
    
    #optimizer 
    
if __name__ == "__main__":
    main()