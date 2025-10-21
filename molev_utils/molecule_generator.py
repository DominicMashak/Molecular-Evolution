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

        validate_molecule(smiles: str, max_atoms: int = 50) -> bool:
            Validates a SMILES string using the mutator.
            
            Args:
                smiles (str): The SMILES string to validate.
                max_atoms (int, optional): Maximum allowed atoms. Defaults to 50.
            
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
        random.seed(seed)
        self.mutator = MoleculeMutator()
        if mutation_weights is None:
            self.mutation_weights = {
                'change_bond': 0.1429,
                'add_atom_inline': 0.1429,
                'add_branch': 0.1428,
                'delete_atom': 0.1429,
                'change_atom': 0.1429,
                'add_ring': 0.1428,
                'delete_ring': 0.1428
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

    def validate_molecule(self, smiles: str, max_atoms: int = 50):
        return self.mutator.validate(smiles, max_atoms)

    def generate_initial_population(self, size: int):
        # Read seeds from initial_seeds.txt
        seeds_file = os.path.join(os.path.dirname(__file__), 'initial_seeds.txt')
        seeds = []
        try:
            with open(seeds_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        seeds.append(line)
        except FileNotFoundError:
            print(f"ERROR: {seeds_file} not found")
        
        # Use dynamically loaded canonicalize_smiles
        global _mol_utils
        if _mol_utils and hasattr(_mol_utils, 'canonicalize_smiles'):
            canonicalize_smiles = _mol_utils.canonicalize_smiles
        else:
            # Fallback: use RDKit canonicalization
            def canonicalize_smiles(smiles):
                from rdkit import Chem
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    return Chem.MolToSmiles(mol, canonical=True)
                return smiles
        
        seeds = [canonicalize_smiles(s) for s in seeds if canonicalize_smiles(s)]
        population = []
        for seed in seeds:
            if self.mutator.validate(seed):
                population.append(seed)
        while len(population) < size:
            base = random.choice(population[:len(seeds)])
            mutant = self.mutate_multiple(base)
            if mutant and mutant not in population:
                population.append(mutant)
        return population[:size]