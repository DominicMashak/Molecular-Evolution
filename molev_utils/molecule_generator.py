import sys
import os
import random
import importlib.util
from pathlib import Path
from collections import defaultdict
from molecule_ops import MoleculeMutator


# Helper function to load canonicalize_smiles
def _load_molecular_utils():
    """load molecular.py from quantum_chemistry."""
    candidates = [
        Path(__file__).parent.parent / 'quantum_chemistry' / 'utils' / 'molecular.py',
        Path.home() / 'Molecular-Evolution' / 'quantum_chemistry' / 'utils' / 'molecular.py',
        Path.cwd() / '..' / 'quantum_chemistry' / 'utils' / 'molecular.py'
    ]
    for p in candidates:
        p = Path(p)
        if p.exists():
            spec = importlib.util.spec_from_file_location("molecular_utils", str(p))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    return None

_mol_utils = _load_molecular_utils()


class MoleculeGenerator:
    """
    A class for generating diverse molecules through mutation-based evolution.

    This class utilizes a shared MoleculeMutator to perform various mutations on SMILES strings,
    allowing for the creation of new molecular structures. It supports adaptive mutation weights
    based on success rates to optimize the mutation process. The generator can produce initial
    populations from seed molecules and validate generated molecules.

    Attributes:
        mutator (MoleculeMutator): An instance of MoleculeMutator for performing mutations.
        mutation_weights (dict): A dictionary mapping mutation types to their normalized weights.
        mutation_stats (defaultdict): Tracks attempts and successes for each mutation type.

    Methods:
        __init__(seed=92, mutation_weights=None):
            Initializes the MoleculeGenerator with a random seed and mutation weights.
            
            Args:
                seed (int, optional): Random seed for reproducibility. Defaults to 92.
                mutation_weights (dict, optional): Custom weights for mutation types. If None,
                    default weights are used. Keys should match mutation type names.

        mutate_multiple(smiles: str, n_mutations: int = None) -> str or None:
            Applies multiple mutations to a SMILES string.
            
            Args:
                smiles (str): The input SMILES string.
                n_mutations (int, optional): Number of mutations to apply. If None, randomly
                    chosen from 1-5 with specified weights.
            
            Returns:
                str or None: The mutated SMILES string if successful, else None.

        mutate(smiles: str) -> str or None:
            Applies a single mutation to a SMILES string with retries.
            
            Args:
                smiles (str): The input SMILES string.
            
            Returns:
                str or None: The mutated SMILES string if valid, else None after max attempts.

        get_mutation_success_rates() -> dict:
            Calculates success rates for each mutation type.
            
            Returns:
                dict: A dictionary with mutation type keys (1-7) and success rate values.

        adapt_mutation_weights(success_rates, adaptation_rate=0.1):
            Adjusts mutation weights based on success rates.
            
            Args:
                success_rates (dict): Success rates from get_mutation_success_rates().
                adaptation_rate (float, optional): Rate of adaptation. Defaults to 0.1.

        validate_molecule(smiles: str, max_atoms: int = 30) -> bool:
            Validates a SMILES string using the mutator.

            Args:
                smiles (str): The SMILES string to validate.
                max_atoms (int, optional): Maximum allowed atoms. Defaults to 30.
            
            Returns:
                bool: True if valid, False otherwise.

        generate_initial_population(size: int) -> list:
            Generates an initial population of SMILES strings from seeds.
            
            Args:
                size (int): Desired population size.
            
            Returns:
                list: A list of unique, valid SMILES strings.
    """
    """Generate diverse molecules using shared MoleculeMutator"""
    def __init__(self, seed=92, mutation_weights=None):
        # Set random seed for reproducibility
        random.seed(seed)

        # Set numpy seed if numpy is available (used by some RDKit operations)
        try:
            import numpy as np
            np.random.seed(seed)
        except ImportError:
            pass

        # Set RDKit random seed
        try:
            from rdkit import rdBase
            rdBase.SeedRandomNumberGenerator(seed)
        except:
            pass

        self.mutator = MoleculeMutator()
        if mutation_weights is None:
            # Equal weights for all mutation types (1/7 each)
            equal_weight = 1.0 / 7.0
            self.mutation_weights = {
                'change_bond': equal_weight,
                'add_atom_inline': equal_weight,
                'add_branch': equal_weight,
                'delete_atom': equal_weight,
                'change_atom': equal_weight,
                'add_ring': equal_weight,
                'delete_ring': equal_weight
            }
        else:
            self.mutation_weights = mutation_weights
        total = sum(self.mutation_weights.values())
        self.mutation_weights = {k: v/total for k, v in self.mutation_weights.items()}
        self.mutation_stats = defaultdict(lambda: {'attempts': 0, 'successes': 0})

    def mutate_multiple(self, smiles: str, n_mutations: int = None):
        if n_mutations is None:
            n_mutations = random.choices([1,2,3,4,5], weights=[0.3,0.3,0.2,0.1,0.1])[0]
        current = smiles
        for _ in range(n_mutations):
            mutated = self.mutate(current)
            if mutated:
                current = mutated
        return current if current != smiles else None

    def mutate(self, smiles: str):
        mutation_types = [1,2,3,4,5,6,7]
        mutation_probs = [self.mutation_weights[k] for k in [
            'change_bond','add_atom_inline','add_branch','delete_atom','change_atom','add_ring','delete_ring']]
        attempts = 0
        max_attempts = 10
        while attempts < max_attempts:
            attempts += 1
            mutation_type = random.choices(mutation_types, mutation_probs)[0]
            self.mutation_stats[mutation_type]['attempts'] += 1
            new_smiles = self.mutator.mutate(smiles, mutation_type)
            if new_smiles and self.mutator.validate(new_smiles):
                self.mutation_stats[mutation_type]['successes'] += 1
                return new_smiles
        return None

    def get_mutation_success_rates(self):
        rates = {}
        for k in [1,2,3,4,5,6,7]:
            stats = self.mutation_stats[k]
            rates[k] = stats['successes']/stats['attempts'] if stats['attempts'] > 0 else 0.0
        return rates

    def adapt_mutation_weights(self, success_rates, adaptation_rate=0.1):
        keys = [
            'change_bond','add_atom_inline','add_branch','delete_atom','change_atom','add_ring','delete_ring'
        ]
        for i, k in enumerate(keys):
            rate = success_rates.get(i+1, 0.0)
            delta = adaptation_rate * (rate - 0.5)
            self.mutation_weights[k] = max(0.05, min(0.5, self.mutation_weights[k] * (1 + delta)))
        total = sum(self.mutation_weights.values())
        self.mutation_weights = {k: v/total for k, v in self.mutation_weights.items()}

    def validate_molecule(self, smiles: str, max_atoms: int = 30):
        return self.mutator.validate(smiles, max_atoms)

    def generate_initial_population(self, size: int, save_to_file=False, seed_number=None, algorithm_name=None):
        """Generate random initial molecules based on random seed.

        Uses simple base molecules and applies multiple mutations to create
        diverse starting populations. The random seed controls both selection
        of base molecules and the mutations applied, ensuring reproducibility.

        Args:
            size (int): Number of molecules to generate
            save_to_file (bool): Whether to save the generated population to a file
            seed_number (int): Random seed used for generation (for filename/metadata)
            algorithm_name (str): Name of algorithm using these seeds (for filename/metadata)

        Returns:
            list: List of valid SMILES strings
        """
        # Simple base molecules (very basic structures to start from)
        base_molecules = [
            'C',           # Methane
            'CC',          # Ethane
            'CCC',         # Propane
            'CCCC',        # Butane
            'CCO',         # Ethanol
            'CCN',         # Ethylamine
            'C=C',         # Ethene
            'C=CC',        # Propene
            'C1CC1',       # Cyclopropane
            'C1CCC1',      # Cyclobutane
            'C1CCCC1',     # Cyclopentane
            'C1CCCCC1',    # Cyclohexane
        ]

        population = []
        attempts = 0
        max_attempts = size * 100  # Prevent infinite loops

        while len(population) < size and attempts < max_attempts:
            attempts += 1

            # Randomly select a base molecule
            base = random.choice(base_molecules)

            # Apply multiple mutations (5-15 mutations to create diversity)
            n_mutations = random.randint(5, 15)
            mutant = base

            for _ in range(n_mutations):
                # Try to mutate
                temp = self.mutate_multiple(mutant, n_mutations=1)
                if temp:
                    mutant = temp

            # Add to population if valid and unique
            if mutant and mutant not in population and self.mutator.validate(mutant):
                population.append(mutant)

        # If we couldn't generate enough, fill with base molecules
        if len(population) < size:
            for base in base_molecules:
                if len(population) >= size:
                    break
                if base not in population and self.mutator.validate(base):
                    population.append(base)

        final_population = population[:size]

        # Save to file if requested
        if save_to_file:
            from datetime import datetime

            # Create generated_seeds directory if it doesn't exist
            seeds_dir = os.path.join(os.path.dirname(__file__), 'generated_seeds')
            os.makedirs(seeds_dir, exist_ok=True)

            # Create filename with algorithm and seed
            algo_str = f"{algorithm_name}_" if algorithm_name else ""
            seed_str = f"seed_{seed_number}_" if seed_number is not None else ""
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"initial_seeds_{algo_str}{seed_str}{timestamp}.txt"
            filepath = os.path.join(seeds_dir, filename)

            # Write to file with metadata
            with open(filepath, 'w') as f:
                f.write("# Initial Seeds for Molecular Evolution\n")
                f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                if algorithm_name:
                    f.write(f"# Algorithm: {algorithm_name}\n")
                if seed_number is not None:
                    f.write(f"# Random Seed: {seed_number}\n")
                f.write(f"# Population Size: {len(final_population)}\n")
                f.write("#\n")
                f.write("# SMILES strings (one per line):\n")
                f.write("#" + "="*60 + "\n\n")

                for i, smiles in enumerate(final_population, 1):
                    f.write(f"{smiles}\n")

            print(f"Initial seeds saved to: {filepath}")

        return final_population
