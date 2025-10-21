import random
import signal
import sys
import os
import argparse
import numpy as np
import multiprocessing
import shutil
from plotting import plot_progress 

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from molev_utils.molecule_ops import mutate_smiles, MoleculeMutator
from molev_utils.io_utils import cleanup_mopac_files, update_progress_file

def get_calculator(calculator_type, **kwargs):
    if calculator_type == "dft":
        from quantum_chemistry.calculators.dft import DFTCalculator
        return DFTCalculator(**kwargs)
    elif calculator_type == "cc":
        from quantum_chemistry.calculators.cc import CCCalculator
        return CCCalculator(**kwargs)
    elif calculator_type == "semiempirical":
        from quantum_chemistry.calculators.semiempirical import SemiEmpiricalCalculator
        return SemiEmpiricalCalculator(**kwargs)
    else:
        raise ValueError(f"Unknown calculator type: {calculator_type}")

# Simulated Annealing parameters
T_initial = 100.0
T_min = 0.01
cooling_rate = 0.95
max_iterations = 100
initial_smiles = "C-CN(=(=N)C-N)-C=C-C=C"
interrupted = False
current_mopac_files = []

def signal_handler(signum, frame):
    """Keyboard interrupt (Ctrl+C)"""
    global interrupted
    print("\n\nInterrupt, cleaning up...")
    interrupted = True
    cleanup_mopac_files(current_mopac_files)
    sys.exit(0)

def get_fitness(smiles, calculator):
    # Convert SMILES to atomic numbers and positions (using RDKit)
    from rdkit import Chem
    from rdkit.Chem import AllChem
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return -1000.0
    mol = Chem.AddHs(mol)
    try:
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        return -1000.0
    conf = mol.GetConformer()
    positions = []
    atomic_numbers = []
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        positions.append([pos.x, pos.y, pos.z])
        atomic_numbers.append(atom.GetAtomicNum())
    positions = np.array(positions)
    atomic_numbers = np.array(atomic_numbers)
    # Run quantum chemistry calculator
    try:
        result = calculator.single_point(atomic_numbers, positions)
        # Use hyperpolarizability beta_mean as fitness (higher is better)
        beta = result.get('beta_mean', -1000.0)
        if beta > 999999:
            beta = -1000.0
        return beta
    except Exception as e:
        print(f"Calculator error: {e}")
        return -1000.0

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
    calculator = get_calculator(calculator_type, **calculator_kwargs)

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

    current_beta = get_fitness(current_smiles, calculator)
    if current_beta is None or current_beta == -1000.0:
        print("Initial molecule invalid in fitness calculation.")
        return

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
            new_beta = get_fitness(new_smiles, calculator)
            if new_beta is None or new_beta == -1000.0:
                mutated_molecules.append((new_smiles, -1000.0))
            else:
                mutated_molecules.append((new_smiles, new_beta))
                print(f"Mutation type {mutation_type}: Beta = {new_beta}")
        
        if interrupted:
            break
        
        # Update progress file with all mutation results
        update_progress_file(progress_file, iteration + 1, T, best_smiles, best_beta, mutated_molecules)
        
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
    
    if not interrupted:
        print(f"\n=== FINAL RESULTS ===")
        print(f"Best SMILES: {best_smiles}")
        print(f"Best Beta: {best_beta:.2f}")
        initial_beta = get_fitness(initial_smiles, calculator)
        print(f"Initial Beta: {initial_beta:.2f}")
        if initial_beta != 0:
            print(f"Improvement factor: {best_beta/initial_beta:.2f}x")
        print(f"Progress saved to: {progress_file}")
    else:
        print("Optimization interrupted by user")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulated Annealing Molecular Optimization")
    parser.add_argument('--calculator', type=str, required=True, help='Calculator type (dft, cc, semiempirical)')
    parser.add_argument('--basis', type=str, default="6-31G", help='Basis set')
    parser.add_argument('--functional', type=str, default="B3LYP", help='Functional (for DFT)')
    parser.add_argument('--method', type=str, default=None, help='Method (for DFT/CC)')
    parser.add_argument('--se-method', type=str, default=None, help='Semiempirical method (PM7, PM6, etc)')
    parser.add_argument('--other', type=str, default=None, help='Other calculator options')
    parser.add_argument('--T_initial', type=float, default=100.0, help='Initial temperature')
    parser.add_argument('--T_min', type=float, default=0.01, help='Minimum temperature')
    parser.add_argument('--cooling_rate', type=float, default=0.95, help='Cooling rate')
    parser.add_argument('--max_iterations', type=int, default=100, help='Number of iterations/generations')
    parser.add_argument('--initial_smiles', type=str, default="CCO", help='Initial SMILES string')
    parser.add_argument('--seed', type=int, default=None, help='Index of predefined seed molecule (0-9), overrides initial_smiles if provided')
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
    if args.se_method:
        calculator_kwargs['method'] = args.se_method
        calculator_kwargs['se_method'] = args.se_method
    calculator_kwargs['n_threads'] = multiprocessing.cpu_count()
    valid_seeds = [
        "CCO", "CCN", "CCC", "C1CCCCC1", "c1ccccc1", "CC(=O)O", "C1COC1", "C1CC1", "C1CCC1", "C1CCCC1"
    ]
    if args.seed is not None and args.initial_smiles == "CCO":  # Only override if initial_smiles is default
        if 0 <= args.seed < len(valid_seeds):
            args.initial_smiles = valid_seeds[args.seed]
        else:
            print(f"Invalid seed {args.seed}, using default initial_smiles")
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