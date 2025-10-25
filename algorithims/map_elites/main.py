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
    
    # Other options
    parser.add_argument('--seed', type=int, default=42, 
                       help='Random seed')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Set random seed
    import random
    random.seed(args.seed)
    
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
    
    # Example: Evolving molecules with beta as objective, num_atoms and homo_lumo_gap as measures
    mutation_weights = {k: 0.1 for k in ['change_bond', 'add_atom_inline', 'add_branch', 'delete_atom', 'change_atom', 'add_ring', 'delete_ring']}
    generator = MoleculeGenerator(mutation_weights=mutation_weights)
    
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
        """Evaluate molecule: fitness is beta_mean, measures are binned num_atoms and homo_lumo_gap."""
        from rdkit import Chem
        
        if solution is None:
            return {
                'beta_mean': 0.0,
                'num_atoms_bin': 0,
                'homo_lumo_bin': 0,
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
                'homo_lumo_bin': 0,
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
        
        # Bin measures for MAP-Elites (0-9)
        num_atoms_bin = min(9, num_atoms // 5)  # Assuming max ~50 atoms, bin size 5
        homo_lumo_bin = min(9, int(homo_lumo_gap))  # Assuming gap in eV, bin by integer
        
        return {
            'beta_mean': beta_mean,
            'num_atoms_bin': num_atoms_bin,
            'homo_lumo_bin': homo_lumo_bin,
            'num_atoms': num_atoms,
            'num_bonds': num_bonds,
            'homo_lumo_gap': homo_lumo_gap,
            'dipole_moment': qc_result.get('dipole_moment', 0.0) or 0.0,
            'total_energy': qc_result.get('total_energy', 0.0) or 0.0,
            'alpha_mean': qc_result.get('alpha_mean', 0.0) or 0.0,
            'gamma': qc_result.get('gamma', 0.0) or 0.0,
            'error': qc_result.get('error')
        }
    
    # Create archive and optimizer
    archive = ar.MAPElitesArchive(
        measure_dims=[10, 10],
        measure_keys=['num_atoms_bin', 'homo_lumo_bin'],
        objective_key='beta_mean'
    )
    
    optimizer = op.MAPElitesOptimizer(
        archive=archive,
        generate_fn=generate_solution,
        mutate_fn=mutate_solution,
        evaluate_fn=evaluate_solution,
        random_init_size=args.pop_size
    )
    
    # Run optimization
    history = optimizer.run(
        n_generations=args.n_gen,
        iterations_per_generation=args.iterations_per_gen,
        log_frequency=args.log_frequency
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