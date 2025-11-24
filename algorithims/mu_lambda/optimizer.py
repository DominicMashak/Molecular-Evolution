"""
(μ+λ) Evolution Strategy for Molecular Optimization
Single-objective evolutionary algorithm where μ parents and λ offspring compete together.
Based on NSGA-II code but simplified for single-objective optimization.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'molev_utils')))

import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from molecule_generator import MoleculeGenerator
from quantum_chemistry_interface import QuantumChemistryInterface

logger = logging.getLogger(__name__)


def _safe_float(x, default=0.0):
    """Convert x to float, handling None and invalid values"""
    if x is None:
        return default
    try:
        return float(x)
    except (ValueError, TypeError):
        return default


@dataclass
class Individual:
    """Represents a single molecule in the population"""
    smiles: str
    fitness: float  # Single objective value
    generation: int
    # Additional properties for tracking
    beta_mean: float = 0.0
    natoms: int = 0
    homo_lumo_gap: float = 0.0
    dipole_moment: float = 0.0
    alpha_mean: float = 0.0
    gamma: float = 0.0
    total_energy: float = 0.0

    def __lt__(self, other):
        """For sorting by fitness (higher is better for maximization)"""
        return self.fitness < other.fitness

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'smiles': self.smiles,
            'fitness': self.fitness,
            'generation': self.generation,
            'beta': self.beta_mean,
            'natoms': self.natoms,
            'homo_lumo_gap': self.homo_lumo_gap,
            'dipole_moment': self.dipole_moment,
            'alpha': self.alpha_mean,
            'gamma': self.gamma,
            'total_energy': self.total_energy
        }


class MuLambdaOptimizer:
    """
    (μ+λ) Evolution Strategy for single-objective molecular optimization

    μ (mu): Number of parents
    λ (lambda): Number of offspring per generation
    '+' means parents and offspring compete together for selection
    """

    def __init__(
        self,
        mu: int,  # Number of parents
        lambda_: int,  # Number of offspring
        n_gen: int,  # Number of generations
        objective_property: str,  # Property to optimize (e.g., 'beta_mean')
        maximize: bool,  # True to maximize, False to minimize
        generator: MoleculeGenerator,
        qc_interface: QuantumChemistryInterface,
        output_dir: str = "mu_lambda_results",
        initial_seeds: List[str] = None,
        save_frequency: int = 10,
        log_frequency: int = 1,
        seed: int = None
    ):
        """
        Initialize (μ+λ) optimizer

        Args:
            mu: Number of parents to maintain
            lambda_: Number of offspring to generate per generation
            n_gen: Number of generations to run
            objective_property: Property name to optimize (e.g., 'beta_mean', 'homo_lumo_gap')
            maximize: True to maximize objective, False to minimize
            generator: MoleculeGenerator instance for mutations
            qc_interface: QuantumChemistryInterface for evaluations
            output_dir: Directory for saving results
            initial_seeds: Optional list of SMILES to seed population
            save_frequency: How often to save population (generations)
            log_frequency: How often to log progress (generations)
        """
        self.mu = mu
        self.lambda_ = lambda_
        self.n_gen = n_gen
        self.objective_property = objective_property
        self.maximize = maximize
        self.generator = generator
        self.qc_interface = qc_interface
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.initial_seeds = initial_seeds or []
        self.save_frequency = save_frequency
        self.log_frequency = log_frequency
        self.seed = seed

        # Population
        self.population: List[Individual] = []
        self.generation = 0

        # Database of all evaluated molecules (for reproducibility)
        self.all_molecules = []

        # Statistics
        self.best_fitness_history = []
        self.mean_fitness_history = []
        self.std_fitness_history = []

        logger.info(f"Initialized (μ+λ) optimizer: μ={mu}, λ={lambda_}")
        logger.info(f"Objective: {objective_property} ({'maximize' if maximize else 'minimize'})")

    def create_individual(self, smiles: str, generation: int) -> Individual:
        """Evaluate a molecule and create an Individual"""
        from rdkit import Chem

        # Calculate quantum properties
        qc_result = self.qc_interface.calculate(smiles)

        # Handle calculation errors
        if qc_result.get('error'):
            logger.warning(f"QC error for {smiles}: {qc_result['error']}")
            fitness = -1e9 if self.maximize else 1e9  # Very bad fitness
            return Individual(
                smiles=smiles,
                fitness=fitness,
                generation=generation
            )

        # Extract fitness value
        fitness = _safe_float(qc_result.get(self.objective_property, 0.0))

        # Get molecule info
        mol = Chem.MolFromSmiles(smiles)
        natoms = mol.GetNumAtoms() if mol else 0

        return Individual(
            smiles=smiles,
            fitness=fitness,
            generation=generation,
            beta_mean=_safe_float(qc_result.get('beta_mean', 0.0)),
            natoms=natoms,
            homo_lumo_gap=_safe_float(qc_result.get('homo_lumo_gap', 0.0)),
            dipole_moment=_safe_float(qc_result.get('dipole_moment', 0.0)),
            alpha_mean=_safe_float(qc_result.get('alpha_mean', 0.0)),
            gamma=_safe_float(qc_result.get('gamma', 0.0)),
            total_energy=_safe_float(qc_result.get('total_energy', 0.0))
        )

    def update_molecule_database(self, individuals: List[Individual]):
        """Update the molecule database with evaluated individuals"""
        for ind in individuals:
            # Check if molecule already exists
            existing = next((m for m in self.all_molecules if m['smiles'] == ind.smiles), None)

            if existing:
                # Update if this is from an earlier generation
                if ind.generation < existing['generation']:
                    existing.update(ind.to_dict())
            else:
                # Add new entry
                self.all_molecules.append(ind.to_dict())

    def save_molecule_database(self):
        """Save the complete molecule database to JSON"""
        db_file = self.output_dir / "all_molecules_database.json"

        # Sort by generation
        sorted_molecules = sorted(self.all_molecules, key=lambda x: x.get('generation', 0))

        with open(db_file, 'w') as f:
            json.dump(sorted_molecules, f, indent=2)

        logger.info(f"Saved {len(sorted_molecules)} molecules to {db_file}")

    def save_population(self, generation: int):
        """Save current population to JSON"""
        pop_file = self.output_dir / f"population_gen_{generation:04d}.json"

        pop_data = {
            'generation': generation,
            'mu': self.mu,
            'lambda': self.lambda_,
            'objective': self.objective_property,
            'maximize': self.maximize,
            'individuals': [ind.to_dict() for ind in self.population]
        }

        with open(pop_file, 'w') as f:
            json.dump(pop_data, f, indent=2)

        logger.info(f"Saved population to {pop_file}")

    def save_metrics(self):
        """Save performance metrics"""
        metrics_file = self.output_dir / "performance_metrics.json"

        metrics = {
            'generations': list(range(len(self.best_fitness_history))),
            'best_fitness': self.best_fitness_history,
            'mean_fitness': self.mean_fitness_history,
            'std_fitness': self.std_fitness_history,
            'objective': self.objective_property,
            'maximize': self.maximize
        }

        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Saved metrics to {metrics_file}")

    def tournament_selection(self, population: List[Individual], k: int = 3) -> Individual:
        """Select an individual using tournament selection"""
        tournament = np.random.choice(population, min(k, len(population)), replace=False).tolist()

        if self.maximize:
            return max(tournament, key=lambda x: x.fitness)
        else:
            return min(tournament, key=lambda x: x.fitness)

    def create_offspring(self, parents: List[Individual]) -> List[Individual]:
        """Create offspring through mutation"""
        offspring = []

        for _ in range(self.lambda_):
            if not parents:
                break

            # Select parent
            parent = self.tournament_selection(parents)

            # Mutate
            try:
                mutated_smiles = self.generator.mutate_multiple(parent.smiles)

                if mutated_smiles and self.generator.validate_molecule(mutated_smiles):
                    child = self.create_individual(mutated_smiles, self.generation + 1)
                    offspring.append(child)
                else:
                    # Mutation failed, try again with simple mutation
                    logger.debug(f"Mutation failed, retrying...")
            except Exception as e:
                logger.debug(f"Mutation exception: {e}")

        return offspring

    def environmental_selection(self, combined: List[Individual]) -> List[Individual]:
        """Select best μ individuals from combined population"""
        # Sort by fitness
        if self.maximize:
            sorted_pop = sorted(combined, key=lambda x: x.fitness, reverse=True)
        else:
            sorted_pop = sorted(combined, key=lambda x: x.fitness)

        # Return top μ
        return sorted_pop[:self.mu]

    def run(self):
        """Main optimization loop"""
        logger.info(f"Starting (μ+λ) optimization")
        logger.info(f"μ={self.mu}, λ={self.lambda_}, generations={self.n_gen}")
        logger.info(f"Objective: {self.objective_property} ({'max' if self.maximize else 'min'})")

        # Initialize population
        logger.info("Generating initial population...")
        if self.initial_seeds:
            initial_smiles = self.initial_seeds[:self.mu]
            if len(initial_smiles) < self.mu:
                additional = self.generator.generate_initial_population(
                    self.mu - len(initial_smiles),
                    save_to_file=True,
                    seed_number=self.seed,
                    algorithm_name="mu_lambda"
                )
                initial_smiles.extend(additional)
        else:
            initial_smiles = self.generator.generate_initial_population(
                self.mu,
                save_to_file=True,
                seed_number=self.seed,
                algorithm_name="mu_lambda"
            )

        logger.info("Evaluating initial population...")
        self.population = []
        for i, smiles in enumerate(initial_smiles):
            logger.info(f"  Evaluating {i+1}/{self.mu}: {smiles}")
            ind = self.create_individual(smiles, 0)
            self.population.append(ind)
            logger.info(f"    {self.objective_property}={ind.fitness:.6e}")

        # Update database
        self.update_molecule_database(self.population)

        # Track initial metrics
        fitnesses = [ind.fitness for ind in self.population]
        self.best_fitness_history.append(max(fitnesses) if self.maximize else min(fitnesses))
        self.mean_fitness_history.append(np.mean(fitnesses))
        self.std_fitness_history.append(np.std(fitnesses))

        logger.info(f"Initial population - Best: {self.best_fitness_history[0]:.6e}, "
                   f"Mean: {self.mean_fitness_history[0]:.6e}")

        # Main evolution loop
        for gen in range(self.n_gen):
            self.generation = gen

            if gen % self.log_frequency == 0:
                logger.info(f"\n{'='*60}")
                logger.info(f"Generation {gen}/{self.n_gen}")

            # Create offspring
            logger.info(f"Creating {self.lambda_} offspring...")
            offspring = self.create_offspring(self.population)
            logger.info(f"  Generated {len(offspring)} valid offspring")

            # Update database with new offspring
            self.update_molecule_database(offspring)

            # Environmental selection: μ+λ
            combined = self.population + offspring
            self.population = self.environmental_selection(combined)

            # Track metrics
            fitnesses = [ind.fitness for ind in self.population]
            best_fitness = max(fitnesses) if self.maximize else min(fitnesses)
            mean_fitness = np.mean(fitnesses)
            std_fitness = np.std(fitnesses)

            self.best_fitness_history.append(best_fitness)
            self.mean_fitness_history.append(mean_fitness)
            self.std_fitness_history.append(std_fitness)

            if gen % self.log_frequency == 0:
                logger.info(f"Best {self.objective_property}: {best_fitness:.6e}")
                logger.info(f"Mean {self.objective_property}: {mean_fitness:.6e}")
                logger.info(f"Std {self.objective_property}: {std_fitness:.6e}")

                # Show best individual
                best_ind = max(self.population, key=lambda x: x.fitness) if self.maximize else min(self.population, key=lambda x: x.fitness)
                logger.info(f"Best molecule: {best_ind.smiles}")

            # Save periodically
            if (gen + 1) % self.save_frequency == 0:
                self.save_population(gen + 1)
                self.save_molecule_database()
                self.save_metrics()

        # Final save
        logger.info("\nOptimization complete!")
        self.save_population(self.n_gen)
        self.save_molecule_database()
        self.save_metrics()

        # Print final statistics
        best_ind = max(self.population, key=lambda x: x.fitness) if self.maximize else min(self.population, key=lambda x: x.fitness)
        logger.info(f"\nFinal best {self.objective_property}: {best_ind.fitness:.6e}")
        logger.info(f"Best molecule: {best_ind.smiles}")
        logger.info(f"Total molecules evaluated: {len(self.all_molecules)}")

        return self.population

    @staticmethod
    def recalculate_from_database(results_dir: str, objective_property: str = None,
                                  maximize: bool = True):
        """
        Recalculate metrics from existing all_molecules_database.json

        Args:
            results_dir: Directory containing all_molecules_database.json
            objective_property: Property to use as fitness (if None, inferred from data)
            maximize: True to maximize, False to minimize
        """
        import json
        from pathlib import Path

        results_path = Path(results_dir)
        db_file = results_path / "all_molecules_database.json"

        if not db_file.exists():
            logger.error(f"Database file not found: {db_file}")
            return

        logger.info(f"Loading molecule database from {db_file}...")
        with open(db_file, 'r') as f:
            molecules = json.load(f)

        logger.info(f"Loaded {len(molecules)} molecules")

        # Find max generation
        max_gen = max((m.get('generation', 0) for m in molecules), default=0)
        logger.info(f"Max generation: {max_gen}")

        # Infer objective if not provided
        if objective_property is None:
            objective_property = 'fitness'
            logger.info(f"Using 'fitness' as objective property")

        # Recalculate metrics for each generation
        metrics = {
            'generations': [],
            'best_fitness': [],
            'mean_fitness': [],
            'std_fitness': [],
            'population_size': []
        }

        for gen in range(max_gen + 1):
            gen_molecules = [m for m in molecules if m.get('generation', 0) <= gen]

            if not gen_molecules:
                continue

            # Get latest generation individuals only (μ best)
            current_gen_mols = [m for m in molecules if m.get('generation', 0) == gen]

            fitnesses = [m.get(objective_property, m.get('fitness', 0.0)) for m in gen_molecules]

            best_fitness = max(fitnesses) if maximize else min(fitnesses)
            mean_fitness = np.mean(fitnesses)
            std_fitness = np.std(fitnesses)

            metrics['generations'].append(gen)
            metrics['best_fitness'].append(float(best_fitness))
            metrics['mean_fitness'].append(float(mean_fitness))
            metrics['std_fitness'].append(float(std_fitness))
            metrics['population_size'].append(len(gen_molecules))

            if gen % 10 == 0:
                logger.info(f"Gen {gen}: Best={best_fitness:.6e}, Mean={mean_fitness:.6e}, "
                          f"Std={std_fitness:.6e}, N={len(gen_molecules)}")

        # Save recalculated metrics
        recalc_dir = results_path / "recalculated"
        recalc_dir.mkdir(exist_ok=True)

        metrics_file = recalc_dir / "performance_metrics_recalculated.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"\nSaved recalculated metrics to {metrics_file}")
        logger.info(f"Final best {objective_property}: {metrics['best_fitness'][-1]:.6e}")
