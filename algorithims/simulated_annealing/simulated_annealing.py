import random
import signal
import sys
import os
import argparse
import numpy as np
import multiprocessing
import shutil
import json
from pathlib import Path
from plotting import plot_progress

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from molev_utils.molecule_ops import mutate_smiles, MoleculeMutator
from molev_utils.io_utils import cleanup_mopac_files, update_progress_file
from molev_utils.quantum_chemistry_interface import QuantumChemistryInterface

# Simulated Annealing parameters
T_initial = 100.0
T_min = 0.01
cooling_rate = 0.95
max_iterations = 100
interrupted = False
current_mopac_files = []

def signal_handler(signum, frame):
    """Keyboard interrupt (Ctrl+C)"""
    global interrupted
    print("\n\nInterrupt, cleaning up...")
    interrupted = True
    cleanup_mopac_files(current_mopac_files)
    sys.exit(0)

def get_fitness(smiles, eval_interface, objective_key='beta_gamma_ratio', maximize=True):
    """
    Calculate fitness and return both fitness value and full result.

    Args:
        smiles: SMILES string to evaluate
        eval_interface: Evaluation interface (QuantumChemistryInterface or SmartCADDInterface)
        objective_key: Property key to use as fitness
        maximize: If True, higher is better; if False, lower is better

    Returns:
        tuple: (fitness_value, result_dict)
    """
    bad_fitness = -1000.0 if maximize else 1000.0
    try:
        result = eval_interface.calculate(smiles)

        if result.get('error'):
            return bad_fitness, result

        fitness = result.get(objective_key, bad_fitness)

        if fitness is None or fitness > 999999 or fitness < -999999:
            fitness = bad_fitness

        return fitness, result
    except Exception as e:
        print(f"Calculator error: {e}")
        import traceback
        traceback.print_exc()
        error_result = {'smiles': smiles, 'error': str(e)}
        return bad_fitness, error_result


def save_all_molecules_database(all_molecules, output_dir):
    """
    Save all evaluated molecules to a JSON database file.

    Args:
        all_molecules: List of molecule dictionaries with QC properties
        output_dir: Directory to save the database
    """
    db_file = Path(output_dir) / "all_molecules_database.json"

    # Convert molecules to serializable format
    serializable_molecules = []
    for mol in all_molecules:
        mol_dict = {
            'smiles': mol.get('smiles', ''),
            'fitness': mol.get('fitness', 0.0),
            'beta': mol.get('beta_mean', 0.0),
            'beta_vec': mol.get('beta_vec'),
            'beta_xxx': mol.get('beta_xxx'),
            'beta_yyy': mol.get('beta_yyy'),
            'beta_zzz': mol.get('beta_zzz'),
            'gamma': mol.get('gamma', 0.0),
            'alpha_mean': mol.get('alpha_mean', 0.0),
            'dipole_moment': mol.get('dipole_moment', 0.0),
            'homo_lumo_gap': mol.get('homo_lumo_gap', 0.0),
            'total_energy': mol.get('total_energy', 0.0),
            'transition_dipole': mol.get('transition_dipole'),
            'oscillator_strength': mol.get('oscillator_strength'),
            'natoms': mol.get('natoms', 0),
            'beta_gamma_ratio': mol.get('beta_gamma_ratio', 0.0),
            'total_energy_atom_ratio': mol.get('total_energy_atom_ratio', 0.0),
            'alpha_range_distance': mol.get('alpha_range_distance', 0.0),
            'homo_lumo_gap_range_distance': mol.get('homo_lumo_gap_range_distance', 0.0),
            'error': mol.get('error')
        }
        serializable_molecules.append(mol_dict)

    with open(db_file, 'w') as f:
        json.dump(serializable_molecules, f, indent=2)

    print(f"Saved {len(serializable_molecules)} molecules to {db_file}")

def simulated_annealing(eval_interface, T_initial, T_min, cooling_rate, max_iterations, initial_smiles,
                        cooling_schedule='exponential', output_dir='sa_results',
                        objective_key='beta_gamma_ratio', maximize=True, atom_set='nlo',
                        generator=None):
    """
    Performs simulated annealing optimization on molecular structures represented as SMILES strings.

    Args:
        eval_interface: Evaluation interface (QuantumChemistryInterface or SmartCADDInterface)
        T_initial: Initial temperature
        T_min: Minimum temperature threshold
        cooling_rate: Cooling rate for exponential cooling
        max_iterations: Maximum number of iterations
        initial_smiles: Starting SMILES string
        cooling_schedule: 'exponential' or 'linear'
        output_dir: Directory to save results
        objective_key: Property key to optimize
        maximize: If True, maximize objective; if False, minimize
        atom_set: Atom set for mutations ('nlo' or 'drug')
    """
    global interrupted
    signal.signal(signal.SIGINT, signal_handler)
    print(f"Starting simulated annealing")
    print(f"Objective: {objective_key} ({'maximize' if maximize else 'minimize'})")
    print("Press Ctrl+C to interrupt and clean up...")

    # Initialize with starting molecule
    mutator = MoleculeMutator()
    valid_seeds = [
        "CCO", "CCN", "CCC", "C1CCCCC1", "c1ccccc1", "CC(=O)O", "C1COC1", "C1CC1", "C1CCC1", "C1CCCC1"
    ]
    current_smiles = initial_smiles
    if not mutator.validate(current_smiles):
        print(f"WARNING: Initial molecule '{current_smiles}' failed validation. Trying fallback seeds.")
        for seed in valid_seeds:
            if mutator.validate(seed):
                print(f"Using fallback seed: {seed}")
                current_smiles = seed
                break
        else:
            print("No valid fallback seed found. Exiting.")
            return

    # Initialize all_molecules database
    all_molecules = []

    bad_fitness = -1000.0 if maximize else 1000.0
    current_beta, current_result = get_fitness(current_smiles, eval_interface, objective_key, maximize)
    if current_beta is None or current_beta == bad_fitness:
        print("Initial molecule invalid in fitness calculation.")
        return

    # Add initial molecule to database
    current_result['fitness'] = current_beta
    all_molecules.append(current_result)

    # Track the best solution found
    best_smiles = current_smiles
    best_beta = current_beta
    
    # Initialize temperature
    T = T_initial
    
    print(f"Initial molecule: {current_smiles}")
    print(f"Initial beta: {current_beta}")
    print(f"Starting temperature: {T}")
    
    # Delete old output directory if it exists
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize progress file
    progress_file = os.path.join(output_dir, "annealing_progress.txt")
    update_progress_file(progress_file, 0, T, best_smiles, best_beta, [])
    
    for iteration in range(max_iterations):
        if interrupted:
            break
            
        print(f"\n--- Iteration {iteration + 1} ---")
        print(f"Current temperature: {T:.4f}")
        
        # Store mutated molecules for this iteration - test all 7 mutation types
        mutated_molecules = []
        
        for mutation_type in range(1, 8):
            if interrupted:
                break

            print(f"\nTesting mutation type {mutation_type}")

            # Generate new candidate solution
            if generator is not None:
                # Use generator's mutate_as_smiles to support SELFIES encoding
                new_smiles = generator.mutate_as_smiles(current_smiles)
                if new_smiles is None:
                    new_smiles = current_smiles
            else:
                new_smiles = mutate_smiles(current_smiles, mutation_type, interrupted)

            if interrupted:
                break

            # Calculate fitness for this mutation
            new_beta, new_result = get_fitness(new_smiles, eval_interface, objective_key, maximize)

            # Add to all_molecules database
            new_result['fitness'] = new_beta
            all_molecules.append(new_result)

            if new_beta is None or new_beta == bad_fitness:
                mutated_molecules.append((new_smiles, bad_fitness))
            else:
                mutated_molecules.append((new_smiles, new_beta))
                print(f"Mutation type {mutation_type}: Beta = {new_beta}")
        
        if interrupted:
            break
        
        # Update progress file with all mutation results
        update_progress_file(progress_file, iteration + 1, T, best_smiles, best_beta, mutated_molecules)

        # Save all_molecules database periodically (every 10 iterations)
        if (iteration + 1) % 10 == 0:
            save_all_molecules_database(all_molecules, output_dir)

        # Plot every iteration
        plot_progress(progress_file, output_dir)
        
        # Select the best mutation from this iteration
        valid_mutations = [(smiles, beta) for smiles, beta in mutated_molecules if beta != bad_fitness]

        if not valid_mutations:
            print(f"Iteration {iteration + 1}: No valid mutations found, keeping current solution")
            T *= cooling_rate
            if T < T_min:
                break
            continue

        # Find the best mutation (direction-aware)
        if maximize:
            best_mutation_smiles, best_mutation_beta = max(valid_mutations, key=lambda x: x[1])
        else:
            best_mutation_smiles, best_mutation_beta = min(valid_mutations, key=lambda x: x[1])

        # Standard Metropolis acceptance criterion
        # delta is positive when mutation is an improvement
        if maximize:
            delta_beta = best_mutation_beta - current_beta
        else:
            delta_beta = current_beta - best_mutation_beta  # Flip for minimization

        if delta_beta > 0:
            # Accept better solution
            print(f"Accepting better solution: {best_mutation_beta} vs {current_beta}")
            current_smiles = best_mutation_smiles
            current_beta = best_mutation_beta

            # Update best solution if this is the best so far
            is_new_best = (best_mutation_beta > best_beta) if maximize else (best_mutation_beta < best_beta)
            if is_new_best:
                best_smiles = best_mutation_smiles
                best_beta = best_mutation_beta
                print(f"New best solution found: {best_beta}")
        else:
            # Probabilistically accept worse solution
            R = random.random()
            acceptance_probability = np.exp(delta_beta / T)
            
            print(f"Worse solution: {best_mutation_beta} < {current_beta}")
            print(f"Acceptance probability: {acceptance_probability:.6f}")
            print(f"Random number: {R:.6f}")
            
            if R < acceptance_probability:
                print("Accepting worse solution")
                current_smiles = best_mutation_smiles
                current_beta = best_mutation_beta
            else:
                print("Rejecting worse solution")
        
        # Update temperature based on schedule
        if cooling_schedule == 'exponential':
            T *= cooling_rate
        elif cooling_schedule == 'linear':
            T = T_initial - (T_initial - T_min) * ((iteration + 1) / max_iterations)
        
        if T < T_min:
            print(f"Temperature {T:.6f} below minimum {T_min}, stopping")
            break
        
        print(f"Current beta: {current_beta:.2f}, Best beta: {best_beta:.2f}")
    
    cleanup_mopac_files(current_mopac_files)

    # Save final all_molecules database
    save_all_molecules_database(all_molecules, output_dir)

    if not interrupted:
        print(f"\n=== FINAL RESULTS ===")
        print(f"Best SMILES: {best_smiles}")
        print(f"Best {objective_key}: {best_beta:.4f}")
        initial_beta, _ = get_fitness(initial_smiles, eval_interface, objective_key, maximize)
        print(f"Initial {objective_key}: {initial_beta:.4f}")
        if initial_beta != 0:
            print(f"Improvement factor: {best_beta/initial_beta:.2f}x")
        print(f"Progress saved to: {progress_file}")
        print(f"All molecules database saved to: {Path(output_dir) / 'all_molecules_database.json'}")
        print(f"Total molecules evaluated: {len(all_molecules)}")
    else:
        print("Optimization interrupted by user")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simulated Annealing Molecular Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Supported Modes:
  qc         - Quantum chemistry (NLO) optimization
  smartcadd  - Drug design optimization via SmartCADD

Examples:
  # NLO: Optimize beta_gamma_ratio using DFT
  python simulated_annealing.py --fitness-mode qc --calculator dft \\
      --functional HF --basis 3-21G --method full_tensor \\
      --objective beta_gamma_ratio --maximize

  # Drug design: Optimize QED using SmartCADD
  python simulated_annealing.py --fitness-mode smartcadd \\
      --smartcadd-mode descriptors --objective qed --maximize
        """
    )

    # Fitness mode
    parser.add_argument('--fitness-mode', type=str, default='qc',
                       choices=['qc', 'smartcadd'],
                       help='Fitness evaluation mode')

    # Objective
    parser.add_argument('--objective', type=str, default='beta_gamma_ratio',
                       help='Property to optimize (beta_gamma_ratio, qed, sa_score, etc.)')
    opt_group = parser.add_mutually_exclusive_group()
    opt_group.add_argument('--maximize', action='store_true', default=True,
                          help='Maximize the objective (default)')
    opt_group.add_argument('--minimize', action='store_true',
                          help='Minimize the objective')

    # Calculator options (qc mode)
    parser.add_argument('--calculator', type=str, required=False,
                       choices=['dft', 'cc', 'semiempirical', 'xtb'],
                       help='Calculator type (required for qc mode)')
    parser.add_argument('--basis', type=str, default="6-31G", help='Basis set')
    parser.add_argument('--functional', type=str, default="B3LYP", help='Functional (for DFT)')
    parser.add_argument('--method', type=str, default=None, help='Method (for DFT/CC)')
    parser.add_argument('--field-strength', type=float, default=0.001, help='Electric field strength')
    parser.add_argument('--se-method', type=str, default=None, help='Semiempirical method (PM7, PM6, etc)')

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

    # SA parameters
    parser.add_argument('--T_initial', type=float, default=100.0, help='Initial temperature')
    parser.add_argument('--T_min', type=float, default=0.01, help='Minimum temperature')
    parser.add_argument('--cooling_rate', type=float, default=0.95, help='Cooling rate')
    parser.add_argument('--max_iterations', type=int, default=100, help='Number of iterations')
    parser.add_argument('--initial_smiles', type=str, default="CCO", help='Initial SMILES string')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--cooling_schedule', type=str, default='exponential',
                       choices=['exponential', 'linear'], help='Temperature cooling schedule')
    parser.add_argument('--output_dir', type=str, default='sa_results', help='Directory to store output files')

    args = parser.parse_args()

    # Set random seeds for reproducibility
    from rdkit import Chem, rdBase

    random.seed(args.seed)
    np.random.seed(args.seed)
    rdBase.SeedRandomNumberGenerator(args.seed)
    print(f"Random seed set to: {args.seed}")

    # Determine atom set
    if args.atom_set:
        atom_set = args.atom_set
    elif args.fitness_mode == 'smartcadd':
        atom_set = 'drug'
    else:
        atom_set = 'nlo'

    # Setup evaluation interface
    if args.fitness_mode == 'smartcadd':
        from molev_utils.smartcadd_interface import SmartCADDInterface
        smartcadd_kwargs = {'mode': args.smartcadd_mode}
        if args.smartcadd_path:
            smartcadd_kwargs['smartcadd_path'] = args.smartcadd_path
        if args.protein_code:
            smartcadd_kwargs['protein_code'] = args.protein_code
        if args.protein_path:
            smartcadd_kwargs['protein_path'] = args.protein_path
        if args.alert_collection:
            smartcadd_kwargs['alert_collection_path'] = args.alert_collection
        eval_interface = SmartCADDInterface(**smartcadd_kwargs)
        print(f"Using SmartCADD evaluation (mode={args.smartcadd_mode})")
    else:
        # Validate calculator is provided for QC mode
        if not args.calculator:
            parser.error("--calculator is required when using --fitness-mode qc")

        calculator_kwargs = {}
        if args.basis:
            calculator_kwargs['basis'] = args.basis
        if args.functional:
            calculator_kwargs['functional'] = args.functional
        if args.se_method:
            calculator_kwargs['se_method'] = args.se_method

        eval_interface = QuantumChemistryInterface(
            calculator_type=args.calculator,
            calculator_kwargs=calculator_kwargs,
            method=args.method,
            field_strength=args.field_strength,
            verbose=False
        )
        print(f"Using quantum chemistry evaluation (calculator={args.calculator})")

    # Determine maximize
    maximize = not args.minimize

    # Generate diverse initial molecule based on random seed
    from molev_utils.molecule_generator import MoleculeGenerator

    generator = MoleculeGenerator(seed=args.seed, atom_set=atom_set, encoding=args.encoding)

    if args.initial_smiles == "CCO":  # Default value - generate diverse molecule
        initial_population = generator.generate_initial_population(size=1, save_to_file=False)
        if initial_population and len(initial_population) > 0:
            # Decode to SMILES (no-op for smiles encoding, decodes SELFIES otherwise)
            initial_smiles = generator.decode_to_smiles(initial_population[0])
            if initial_smiles is None:
                initial_smiles = args.initial_smiles
                print(f"Failed to decode initial molecule, using default: {initial_smiles}")
            else:
                args.initial_smiles = initial_smiles
                print(f"Generated diverse initial molecule for seed {args.seed}: {args.initial_smiles}")
        else:
            print(f"Failed to generate molecule, using default: {args.initial_smiles}")
    else:
        print(f"Using user-specified initial molecule: {args.initial_smiles}")

    simulated_annealing(
        eval_interface=eval_interface,
        T_initial=args.T_initial,
        T_min=args.T_min,
        cooling_rate=args.cooling_rate,
        max_iterations=args.max_iterations,
        initial_smiles=args.initial_smiles,
        cooling_schedule=args.cooling_schedule,
        output_dir=args.output_dir,
        objective_key=args.objective,
        maximize=maximize,
        atom_set=atom_set,
        generator=generator
    )