import random
from typing import Callable, Dict, Any, Optional, List
from archive import MAPElitesArchive


class MAPElitesOptimizer:
    """
    General-purpose MAP-Elites optimizer that can evolve any type of solution.
    Delegates solution generation, mutation, and evaluation to user-provided functions.
    """
    
    def __init__(
        self,
        archive: MAPElitesArchive,
        generate_fn: Callable[[], Any],
        mutate_fn: Callable[[Any], Any],
        evaluate_fn: Callable[[Any], Dict[str, Any]],
        random_init_size: int = 100
    ):
        """
        Initialize the MAP-Elites optimizer.
        
        Args:
            archive: MAPElitesArchive instance for storing solutions
            generate_fn: Function that generates a new random solution.
                        Returns: solution object
            mutate_fn: Function that mutates an existing solution.
                      Args: parent solution
                      Returns: mutated solution object
            evaluate_fn: Function that evaluates a solution and returns its properties.
                        Args: solution object
                        Returns: dict with objective, measures, and any other properties
            random_init_size: Number of random solutions to generate during initialization
        """
        self.archive = archive
        self.generate_fn = generate_fn
        self.mutate_fn = mutate_fn
        self.evaluate_fn = evaluate_fn
        self.random_init_size = random_init_size
        
        # Statistics
        self.total_evaluations = 0
        self.generation = 0
    
    def initialize(self) -> None:
        """
        Initialize the archive with random solutions.
        """
        print(f"Initializing with {self.random_init_size} random solutions...")
        
        for i in range(self.random_init_size):
            # Generate and evaluate a random solution
            solution = self.generate_fn()
            properties = self.evaluate_fn(solution)
            self.total_evaluations += 1
            
            # Try to add to archive
            self.archive.add(solution, properties)
            
            if (i + 1) % max(1, self.random_init_size // 10) == 0:
                print(f"  {i + 1}/{self.random_init_size} - "
                      f"Coverage: {self.archive.get_coverage():.2%}")
        
        print(f"Initialization complete. Archive size: {len(self.archive)}")
    
    def sample_parent(self) -> Optional[Any]:
        """
        Sample a parent solution from the archive uniformly at random.
        
        Returns:
            A solution from the archive, or None if archive is empty
        """
        all_solutions = self.archive.get_all_solutions()
        
        if not all_solutions:
            return None
        
        entry = random.choice(all_solutions)
        return entry['solution']
    
    def step(self, n_iterations: int = 1) -> Dict[str, Any]:
        """
        Run one or more iterations of the MAP-Elites algorithm.
        Each iteration: sample parent -> mutate -> evaluate -> add to archive
        
        Args:
            n_iterations: Number of iterations to run
        
        Returns:
            Dictionary with statistics about this step
        """
        n_added = 0
        n_improvements = 0
        
        for _ in range(n_iterations):
            # Sample a parent from the archive
            parent = self.sample_parent()
            
            if parent is None:
                # Archive is empty, generate random solution
                solution = self.generate_fn()
            else:
                # Mutate the parent
                solution = self.mutate_fn(parent)
            
            # Evaluate the solution
            properties = self.evaluate_fn(solution)
            self.total_evaluations += 1
            
            # Try to add to archive
            was_added = self.archive.add(solution, properties)
            
            if was_added:
                n_added += 1
        
        self.generation += 1
        
        return {
            'generation': self.generation,
            'n_added': n_added,
            'archive_size': len(self.archive),
            'coverage': self.archive.get_coverage(),
            'max_objective': self.archive.get_max_objective(),
            'mean_objective': self.archive.get_mean_objective(),
            'total_evaluations': self.total_evaluations
        }
    
    def run(self, n_generations: int, iterations_per_generation: int = 1, 
            log_frequency: int = 10) -> List[Dict[str, Any]]:
        """
        Run the MAP-Elites algorithm for multiple generations.
        
        Args:
            n_generations: Number of generations to run
            iterations_per_generation: Number of mutations per generation
            log_frequency: How often to print progress (in generations)
        
        Returns:
            List of statistics dictionaries, one per generation
        """
        if len(self.archive) == 0:
            print("Archive is empty. Running initialization first...")
            self.initialize()
        
        print(f"\nRunning MAP-Elites for {n_generations} generations "
              f"({iterations_per_generation} iterations/gen)...")
        
        history = []
        
        for gen in range(n_generations):
            stats = self.step(iterations_per_generation)
            history.append(stats)
            
            if (gen + 1) % log_frequency == 0:
                print(f"Gen {stats['generation']:4d}: "
                      f"Coverage={stats['coverage']:6.2%}, "
                      f"Size={stats['archive_size']:5d}, "
                      f"Max={stats['max_objective']:8.3f}, "
                      f"Mean={stats['mean_objective']:8.3f}, "
                      f"Added={stats['n_added']:3d}/{iterations_per_generation}")
        
        print("\nOptimization complete!")
        print(f"Final coverage: {self.archive.get_coverage():.2%}")
        print(f"Final archive size: {len(self.archive)}")
        print(f"Total evaluations: {self.total_evaluations}")
        
        return history
    
    def get_best_solution(self) -> Optional[Dict[str, Any]]:
        """
        Get the solution with the highest objective value.
        
        Returns:
            Dictionary with solution info, or None if archive is empty
        """
        all_solutions = self.archive.get_all_solutions()
        
        if not all_solutions:
            return None
        
        return max(all_solutions, key=lambda x: x['objective'])


