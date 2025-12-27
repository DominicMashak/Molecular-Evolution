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

def get_fitness(smiles, qc_interface):
    """
    Calculate fitness and return both fitness value and full QC result.

    Returns:
        tuple: (fitness_value, qc_result_dict)
    """
    try:
        print(f"DEBUG: Getting fitness for: {smiles}")
        result = qc_interface.calculate(smiles)
        print(f"DEBUG: QC interface returned result with keys: {result.keys() if result else 'None'}")

        if result.get('error'):
            print(f"DEBUG: Calculation failed with error: {result['error']}")
            return -1000.0, result

        # Use beta_gamma_ratio as fitness (higher is better)
        beta_gamma_ratio = result.get('beta_gamma_ratio', -1000.0)
        print(f"DEBUG: beta_gamma_ratio = {beta_gamma_ratio}")

        if beta_gamma_ratio is None or beta_gamma_ratio > 999999 or beta_gamma_ratio < -999999:
            beta_gamma_ratio = -1000.0

        return beta_gamma_ratio, result
    except Exception as e:
        print(f"Calculator error: {e}")
        import traceback
        traceback.print_exc()
        error_result = {'smiles': smiles, 'error': str(e)}
        return -1000.0, error_result


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

def simulated_annealing(calculator_type, calculator_kwargs, T_initial, T_min, cooling_rate, max_iterations, initial_smiles, cooling_schedule='exponential', output_dir='sa_results'):
    """
    Performs simulated annealing optimization on molecular structures represented as SMILES strings.
    This function implements a simulated annealing algorithm to evolve molecular structures by iteratively
    mutating the current molecule and accepting or rejecting changes based on a fitness function (beta)
    and the Metropolis acceptance criterion. It supports different cooling schedules and calculators
    for fitness evaluation, with progress tracking and output to files.
    Parameters:
    -----------
    calculator_type : str
        The type of calculator to use for fitness evaluation (e.g., 'mopac', 'rdkit'). Must be supported
        by the get_calculator function.
    calculator_kwargs : dict
        Keyword arguments to pass to the calculator initialization.
    T_initial : float
        The initial temperature for the simulated annealing process. Higher values allow more
        exploration of worse solutions early on.
    T_min : float
        The minimum temperature threshold. The algorithm stops when temperature drops below this value.
    cooling_rate : float
        The cooling rate for exponential cooling (multiplicative factor applied to temperature each
        iteration). Ignored if cooling_schedule is 'linear'.
    max_iterations : int
        The maximum number of iterations to perform.
    initial_smiles : str
        The SMILES string of the initial molecule to start the optimization from.
    cooling_schedule : str, optional
        The cooling schedule to use. Options are 'exponential' (default) or 'linear'. For 'linear',
        temperature decreases linearly from T_initial to T_min over max_iterations.
    output_dir : str, optional
        The directory to save progress files and plots. Defaults to 'sa_results'. Existing directory
        is deleted and recreated.
    Notes:
    ------
    - The function handles SIGINT (Ctrl+C) for graceful interruption and cleanup.
    - Molecules are mutated using 7 different mutation types, and the best valid mutation is selected
      each iteration.
    - Fitness (beta) is calculated using the specified calculator; invalid molecules return -1000.0.
    - Progress is saved to 'annealing_progress.txt' in output_dir, and plots are generated each iteration.
    - If the initial SMILES is invalid, fallback seeds are tried.
    - No return value; results are printed and saved to files.
    """
    global interrupted
    signal.signal(signal.SIGINT, signal_handler)
    print("Starting simulated annealing with calculator:", calculator_type)
    print("Press Ctrl+C to interrupt and clean up...")

    # Extract method and field_strength from calculator_kwargs
    method = calculator_kwargs.pop('method', None)
    field_strength = calculator_kwargs.pop('field_strength', 0.001)

    # Initialize QuantumChemistryInterface
    qc_interface = QuantumChemistryInterface(
        calculator_type=calculator_type,
        calculator_kwargs=calculator_kwargs,
        method=method,
        field_strength=field_strength,
        verbose=False
    )

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

    current_beta, current_qc_result = get_fitness(current_smiles, qc_interface)
    if current_beta is None or current_beta == -1000.0:
        print("Initial molecule invalid in fitness calculation.")
        return

    # Add initial molecule to database
    current_qc_result['fitness'] = current_beta
    all_molecules.append(current_qc_result)

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

            # Generate new candidate solution (guaranteed to be valid)
            new_smiles = mutate_smiles(current_smiles, mutation_type, interrupted)

            if interrupted:
                break

            # Calculate beta for this mutation
            new_beta, new_qc_result = get_fitness(new_smiles, qc_interface)

            # Add to all_molecules database
            new_qc_result['fitness'] = new_beta
            all_molecules.append(new_qc_result)

            if new_beta is None or new_beta == -1000.0:
                mutated_molecules.append((new_smiles, -1000.0))
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
        valid_mutations = [(smiles, beta) for smiles, beta in mutated_molecules if beta != -1000.0]
        
        if not valid_mutations:
            print(f"Iteration {iteration + 1}: No valid mutations found, keeping current solution")
            T *= cooling_rate
            if T < T_min:
                break
            continue
        
        # Find the best mutation
        best_mutation_smiles, best_mutation_beta = max(valid_mutations, key=lambda x: x[1])
        
        # Standard Metropolis acceptance criterion
        delta_beta = best_mutation_beta - current_beta
        
        if delta_beta > 0:
            # Accept better solution
            print(f"Accepting better solution: {best_mutation_beta} > {current_beta}")
            current_smiles = best_mutation_smiles
            current_beta = best_mutation_beta
            
            # Update best solution if this is the best so far
            if best_mutation_beta > best_beta:
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
        print(f"Best Beta (beta_gamma_ratio): {best_beta:.2f}")
        initial_beta, _ = get_fitness(initial_smiles, qc_interface)
        print(f"Initial Beta: {initial_beta:.2f}")
        if initial_beta != 0:
            print(f"Improvement factor: {best_beta/initial_beta:.2f}x")
        print(f"Progress saved to: {progress_file}")
        print(f"All molecules database saved to: {Path(output_dir) / 'all_molecules_database.json'}")
        print(f"Total molecules evaluated: {len(all_molecules)}")
    else:
        print("Optimization interrupted by user")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulated Annealing Molecular Optimization")
    parser.add_argument('--calculator', type=str, required=True, help='Calculator type (dft, cc, semiempirical)')
    parser.add_argument('--basis', type=str, default="6-31G", help='Basis set')
    parser.add_argument('--functional', type=str, default="B3LYP", help='Functional (for DFT)')
    parser.add_argument('--method', type=str, default=None, help='Method (for DFT/CC)')
    parser.add_argument('--field-strength', type=float, default=0.001, help='Electric field strength for hyperpolarizability calculations')
    parser.add_argument('--se-method', type=str, default=None, help='Semiempirical method (PM7, PM6, etc)')
    parser.add_argument('--other', type=str, default=None, help='Other calculator options')
    parser.add_argument('--T_initial', type=float, default=100.0, help='Initial temperature')
    parser.add_argument('--T_min', type=float, default=0.01, help='Minimum temperature')
    parser.add_argument('--cooling_rate', type=float, default=0.95, help='Cooling rate')
    parser.add_argument('--max_iterations', type=int, default=100, help='Number of iterations/generations')
    parser.add_argument('--initial_smiles', type=str, default="CCO", help='Initial SMILES string')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--cooling_schedule', type=str, default='exponential', choices=['exponential', 'linear'], help='Temperature cooling schedule')
    parser.add_argument('--output_dir', type=str, default='sa_results', help='Directory to store output files and plots')
    args = parser.parse_args()
    calculator_kwargs = {}
    if args.basis:
        calculator_kwargs['basis'] = args.basis
    if args.functional:
        calculator_kwargs['functional'] = args.functional
    if args.method:
        calculator_kwargs['method'] = args.method
    if args.field_strength:
        calculator_kwargs['field_strength'] = args.field_strength
    if args.se_method:
        calculator_kwargs['method'] = args.se_method
        calculator_kwargs['se_method'] = args.se_method
    calculator_kwargs['n_threads'] = multiprocessing.cpu_count()

    # Set random seeds for reproducibility
    import numpy as np
    from rdkit import Chem, rdBase

    random.seed(args.seed)
    np.random.seed(args.seed)
    rdBase.SeedRandomNumberGenerator(args.seed)
    print(f"Random seed set to: {args.seed}")

    # Generate diverse initial molecule based on random seed
    # This ensures different seed values start from different molecules for better exploration
    # Only auto-generate if initial_smiles is still the default "CCO"
    from molev_utils.molecule_generator import MoleculeGenerator

    if args.initial_smiles == "CCO":  # Default value - generate diverse molecule
        generator = MoleculeGenerator(seed=args.seed)
        initial_population = generator.generate_initial_population(size=1, save_to_file=False)
        if initial_population and len(initial_population) > 0:
            args.initial_smiles = initial_population[0]
            print(f"Generated diverse initial molecule for seed {args.seed}: {args.initial_smiles}")
        else:
            print(f"Failed to generate molecule, using default: {args.initial_smiles}")
    else:  # User explicitly provided a molecule
        print(f"Using user-specified initial molecule: {args.initial_smiles}")

    simulated_annealing(
        args.calculator,
        calculator_kwargs,
        T_initial=args.T_initial,
        T_min=args.T_min,
        cooling_rate=args.cooling_rate,
        max_iterations=args.max_iterations,
        initial_smiles=args.initial_smiles,
        cooling_schedule=args.cooling_schedule,
        output_dir=args.output_dir
    )