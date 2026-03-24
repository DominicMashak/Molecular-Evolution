import sys
import os
import random
import importlib.util
from pathlib import Path
from collections import defaultdict
from molev_utils.molecule_ops import MoleculeMutator

# Lazy-import SELFIES support — only loaded when encoding='selfies' is requested.
# This keeps the default SMILES path free of the selfies dependency.
_SELFIESMutator = None
_smiles_to_selfies = None
_selfies_to_smiles = None

def _load_selfies_ops():
    """Import selfies_ops on first use; raises ImportError with a clear message if selfies is not installed."""
    global _SELFIESMutator, _smiles_to_selfies, _selfies_to_smiles
    if _SELFIESMutator is None:
        try:
            from molev_utils.selfies_ops import SELFIESMutator, smiles_to_selfies, selfies_to_smiles
            _SELFIESMutator = SELFIESMutator
            _smiles_to_selfies = smiles_to_selfies
            _selfies_to_smiles = selfies_to_smiles
        except ImportError:
            raise ImportError(
                "The 'selfies' package is required for SELFIES encoding. "
                "Install it with: pip install selfies"
            )


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
    def __init__(self, seed=92, mutation_weights=None, atom_set='nlo', encoding='smiles'):
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

        if encoding not in ('smiles', 'selfies'):
            raise ValueError(f"Unknown encoding '{encoding}'. Must be 'smiles' or 'selfies'.")
        self.encoding = encoding
        self.atom_set = atom_set
        self.mutator = MoleculeMutator(atom_set=atom_set)
        if encoding == 'selfies':
            _load_selfies_ops()
            self.selfies_mutator = _SELFIESMutator(atom_set=atom_set)
        else:
            self.selfies_mutator = None
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
        self._crossover = None  # lazily initialised by crossover_as_smiles()

    def crossover_in_encoding(self, sol1: str, sol2: str) -> 'str | None':
        """Fragment-based crossover, returning the result in the configured encoding.

        For MOME / MAP-Elites where the archive stores solutions in the internal
        encoding (SMILES or SELFIES), this ensures the offspring is in the same
        format as the parents so it can be stored in the archive without
        additional conversion steps.
        """
        result_smiles = self.crossover_as_smiles(sol1, sol2)
        if result_smiles is None:
            return None
        if self.encoding == 'selfies':
            _load_selfies_ops()
            return _smiles_to_selfies(result_smiles)
        return result_smiles

    def crossover_as_smiles(self, smiles1: str, smiles2: str) -> 'str | None':
        """Fragment-based crossover on two SMILES strings; always returns SMILES or None.

        When encoding='selfies', the inputs are decoded to SMILES before crossing.
        The crossover engine operates on RDKit mol objects and is encoding-agnostic.
        """
        if self.encoding == 'selfies':
            s1 = self.decode_to_smiles(smiles1)
            s2 = self.decode_to_smiles(smiles2)
        else:
            s1, s2 = smiles1, smiles2
        if s1 is None or s2 is None:
            return None
        if self._crossover is None:
            from molev_utils.crossover_ops import MoleculeCrossover
            self._crossover = MoleculeCrossover(atom_set=self.atom_set)
        return self._crossover.crossover(s1, s2)

    def mutate_as_smiles(self, smiles: str) -> 'str | None':
        """Mutate a SMILES string and always return a SMILES string.

        For 'smiles' encoding this is equivalent to mutate_multiple().
        For 'selfies' encoding the input SMILES is encoded to SELFIES, mutated
        via token-level operations, then decoded back to SMILES.
        """
        if self.encoding == 'smiles':
            return self.mutate_multiple(smiles)
        # SELFIES mode: encode → mutate tokens → decode back to SMILES
        selfies_str = _smiles_to_selfies(smiles)
        if selfies_str is None:
            return None
        mutated_selfies = self._mutate_selfies(selfies_str)
        if mutated_selfies is None:
            return None
        return _selfies_to_smiles(mutated_selfies)

    def validate_as_smiles(self, smiles: str, max_atoms: int = 30) -> bool:
        """Validate a SMILES string using the underlying SMILES validator.

        Always interprets the input as SMILES, regardless of the configured
        encoding.  Useful for NSGA-II and other callers that always work
        with SMILES-based Individual objects.
        """
        return self.mutator.validate(smiles, max_atoms)

    def decode_to_smiles(self, solution: str) -> 'str | None':
        """Convert a solution string to SMILES regardless of encoding.

        For 'smiles' encoding this is a no-op; for 'selfies' it decodes via the
        SELFIES library.  Returns None if decoding fails.
        """
        if self.encoding == 'selfies':
            return _selfies_to_smiles(solution)
        return solution

    def mutate_multiple(self, solution: str, n_mutations: int = None):
        if n_mutations is None:
            n_mutations = random.choices([1,2,3,4,5], weights=[0.3,0.3,0.2,0.1,0.1])[0]
        current = solution
        for _ in range(n_mutations):
            mutated = self.mutate(current)
            if mutated:
                current = mutated
        return current if current != solution else None

    def mutate(self, solution: str):
        if self.encoding == 'selfies':
            return self._mutate_selfies(solution)
        return self._mutate_smiles(solution)

    def _mutate_smiles(self, smiles: str):
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

    def _mutate_selfies(self, selfies_str: str):
        # SELFIES mutations: types 1 (substitute), 2 (insert), 3 (delete)
        selfies_mutation_types = [1, 2, 3]
        attempts = 0
        max_attempts = 10
        while attempts < max_attempts:
            attempts += 1
            mutation_type = random.choice(selfies_mutation_types)
            new_selfies = self.selfies_mutator.mutate(selfies_str, mutation_type)
            if new_selfies and self.selfies_mutator.validate(new_selfies):
                return new_selfies
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

    def validate_molecule(self, solution: str, max_atoms: int = 30):
        if self.encoding == 'selfies':
            return self.selfies_mutator.validate(solution, max_atoms)
        return self.mutator.validate(solution, max_atoms)

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
        # Select base molecules based on atom set
        if self.atom_set == 'drug':
            base_molecules = [
                'c1ccccc1',        # Benzene
                'c1ccncc1',        # Pyridine
                'c1ccoc1',         # Furan
                'c1cc[nH]c1',      # Pyrrole
                'c1ccsc1',         # Thiophene
                'c1ccc2[nH]ccc2c1',  # Indole
                'c1ccc2ncccc2c1',  # Quinoline
                'C1CCNCC1',        # Piperidine
                'C1CNCCN1',        # Piperazine
                'C1CCOCC1',        # Tetrahydropyran (morpholine-like)
                'c1ccc(O)cc1',     # Phenol
                'c1ccc(N)cc1',     # Aniline
                'c1ccc(F)cc1',     # Fluorobenzene
                'c1ccc(Cl)cc1',    # Chlorobenzene
                'c1ccc(Br)cc1',    # Bromobenzene
                'c1ccc(S)cc1',     # Thiophenol
                'c1cnc2ccccc2n1',  # Quinazoline
                'c1ccc(-c2ccccc2)cc1',  # Biphenyl
                'c1nc2ccccc2[nH]1',  # Benzimidazole
                'CC(=O)Nc1ccccc1', # Acetanilide
            ]
        else:
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

        # Convert base molecules to SELFIES encoding if needed
        if self.encoding == 'selfies':
            converted = []
            for smi in base_molecules:
                enc = _smiles_to_selfies(smi)
                if enc:
                    converted.append(enc)
            if converted:
                base_molecules = converted

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
            if mutant and mutant not in population and self.validate_molecule(mutant):
                population.append(mutant)

        # If we couldn't generate enough, fill with base molecules
        if len(population) < size:
            for base in base_molecules:
                if len(population) >= size:
                    break
                if base not in population and self.validate_molecule(base):
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
                enc_label = "SELFIES" if self.encoding == 'selfies' else "SMILES"
                f.write(f"# {enc_label} strings (one per line):\n")
                f.write("#" + "="*60 + "\n\n")

                for i, smiles in enumerate(final_population, 1):
                    f.write(f"{smiles}\n")

            print(f"Initial seeds saved to: {filepath}")

        return final_population
