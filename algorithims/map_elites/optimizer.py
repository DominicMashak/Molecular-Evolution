import random
from typing import Callable, Dict, Any, Optional, List
from archive import MAPElitesArchive
from performance import PerformanceTracker
from plotting import PerformancePlotter


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
        random_init_size: int = 100,
        output_dir: str = "map_elites_results",
        reference_point: List[float] = None
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
            output_dir: Directory for saving results (archives, plots, etc.)
            reference_point: Reference point for hypervolume calculation (list of floats)
        """
        self.archive = archive
        self.generate_fn = generate_fn
        self.mutate_fn = mutate_fn
        self.evaluate_fn = evaluate_fn
        self.random_init_size = random_init_size
        
        from pathlib import Path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.performance_tracker = PerformanceTracker(self.output_dir, reference_point=reference_point)
        self.performance_plotter = PerformancePlotter(self.output_dir)
        
        # Statistics
        self.total_evaluations = 0
        self.generation = 0

        # Database of all evaluated molecules (for reproducibility and post-hoc analysis)
        self.all_molecules = []
    
    def initialize(self) -> None:
        """
        Initialize the archive with random solutions.
        """
        print(f"Initializing with {self.random_init_size} random solutions...")
        
        bin_stats = {}  # Track which bins are being filled
        
        for i in range(self.random_init_size):
            # Generate and evaluate a random solution
            solution = self.generate_fn()
            if solution is None:
                continue  # Skip invalid generations
            properties = self.evaluate_fn(solution)
            self.total_evaluations += 1

            # Update molecule database (generation 0 for initialization)
            self.update_molecule_database(solution, properties, generation=0)

            # Track bin statistics
            bin_key = (properties.get('num_atoms_bin', -1), properties.get('num_bonds_bin', -1))
            bin_stats[bin_key] = bin_stats.get(bin_key, 0) + 1
            
            # Try to add to archive
            was_added = self.archive.add(solution, properties)
            
            if (i + 1) % max(1, self.random_init_size // 10) == 0:
                print(f"  {i + 1}/{self.random_init_size} - "
                      f"Coverage: {self.archive.get_coverage():.2%}, "
                      f"Unique bins tried: {len(bin_stats)}")
        
        print(f"Initialization complete. Archive size: {len(self.archive)}")
        print(f"Bin distribution during init: {len(bin_stats)} unique bins explored")
        
        # Show top 5 most common bins
        sorted_bins = sorted(bin_stats.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"Most common bins: {sorted_bins}")
    
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

    def update_molecule_database(self, solution: Any, properties: Dict[str, Any], generation: int):
        """
        Update the molecule database with a new evaluated molecule.
        If molecule already exists, update with earliest generation it appeared.

        Args:
            solution: The solution (typically a SMILES string)
            properties: Dictionary containing objective, measures, and other properties
            generation: Generation number when this molecule was evaluated
        """
        # Check if molecule already exists
        existing = next((m for m in self.all_molecules if m['smiles'] == solution), None)

        if existing:
            # Update if this generation is earlier
            if generation < existing['generation']:
                existing['generation'] = generation
                existing['objective'] = properties.get('objective')
                # Update all other properties
                for key, value in properties.items():
                    if key != 'smiles':
                        existing[key] = value
        else:
            # Add new molecule entry
            mol_entry = {
                'smiles': solution,
                'generation': generation,
                'objective': properties.get('objective')
            }
            # Add all other properties from evaluation
            for key, value in properties.items():
                if key not in mol_entry:
                    mol_entry[key] = value

            self.all_molecules.append(mol_entry)

    def save_molecule_database(self):
        """
        Save the complete molecule database to JSON file.
        Enables reproducibility and post-hoc analysis without re-evaluation.
        """
        import json
        db_file = self.output_dir / "all_molecules_database.json"

        # Sort by generation for easier analysis
        sorted_molecules = sorted(self.all_molecules, key=lambda x: x.get('generation', 0))

        with open(db_file, 'w') as f:
            json.dump(sorted_molecules, f, indent=2)

        print(f"Saved {len(sorted_molecules)} molecules to {db_file}")

    @staticmethod
    def recalculate_from_database(results_dir: str, archive_config: Dict = None):
        """
        Recalculate archive, QD score, and other metrics from existing all_molecules_database.json.
        Useful for post-hoc analysis without re-running expensive evaluations.

        Args:
            results_dir: Directory containing all_molecules_database.json
            archive_config: Dictionary with archive configuration:
                          - measure_dims: List[int]
                          - measure_keys: List[str]
                          - objective_key: str (default: 'objective')

        This will create new files with recalculated metrics.
        """
        import json
        from pathlib import Path
        from archive import MAPElitesArchive
        from performance import PerformanceTracker
        from plotting import PerformancePlotter

        results_path = Path(results_dir)
        db_file = results_path / "all_molecules_database.json"

        if not db_file.exists():
            raise FileNotFoundError(f"Database file not found: {db_file}")

        print(f"Loading molecule database from {db_file}...")
        with open(db_file, 'r') as f:
            molecules = json.load(f)

        print(f"Loaded {len(molecules)} molecules from database")

        # Use provided config or try to infer from data
        if archive_config is None:
            # Try to infer from first molecule
            if molecules:
                first_mol = molecules[0]
                measure_keys = [k for k in first_mol.keys()
                              if '_bin' in k and k not in ['generation', 'objective']]
                measure_dims = [10, 10]  # Default assumption
                objective_key = 'objective'
                print(f"Inferred measure keys: {measure_keys}")
            else:
                raise ValueError("No molecules in database and no archive_config provided")
        else:
            measure_dims = archive_config['measure_dims']
            measure_keys = archive_config['measure_keys']
            objective_key = archive_config.get('objective_key', 'objective')

        # Create archive and performance tracker
        archive = MAPElitesArchive(
            measure_dims=measure_dims,
            measure_keys=measure_keys,
            objective_key=objective_key
        )

        recalc_dir = results_path / "recalculated"
        recalc_dir.mkdir(exist_ok=True)

        performance_tracker = PerformanceTracker(recalc_dir, reference_point=None)
        performance_plotter = PerformancePlotter(recalc_dir)

        # Group molecules by generation
        max_gen = max(m['generation'] for m in molecules)
        print(f"Rebuilding archive generation by generation (0 to {max_gen})...")

        for gen in range(max_gen + 1):
            # Add all molecules up to this generation
            gen_molecules = [m for m in molecules if m['generation'] <= gen]

            # Rebuild archive from scratch for this generation
            archive_snapshot = MAPElitesArchive(
                measure_dims=measure_dims,
                measure_keys=measure_keys,
                objective_key=objective_key
            )

            for mol in gen_molecules:
                archive_snapshot.add(mol['smiles'], mol)

            # Update performance tracker
            performance_tracker.update(gen, archive_snapshot)

            if gen % 10 == 0 or gen == max_gen:
                print(f"  Gen {gen}: Coverage={archive_snapshot.get_coverage():.2%}, "
                      f"Size={len(archive_snapshot)}")

        # Save recalculated results
        print("\nSaving recalculated results...")
        performance_tracker.save()
        performance_plotter.plot_convergence(performance_tracker)
        performance_plotter.plot_archive_heatmap(archive_snapshot)
        performance_plotter.plot_archive_cells_evolution(performance_tracker)
        performance_plotter.plot_qd_score_evolution(performance_tracker)

        print(f"\nRecalculation complete! Results saved to {recalc_dir}")
        print(f"Final coverage: {archive_snapshot.get_coverage():.2%}")
        print(f"Final archive size: {len(archive_snapshot)}")

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
            
            if solution is None:
                continue  # Skip invalid mutations
            
            # Evaluate the solution
            properties = self.evaluate_fn(solution)
            self.total_evaluations += 1

            # Update molecule database
            self.update_molecule_database(solution, properties, generation=self.generation + 1)

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
            log_frequency: int = 10, save_frequency: int = 50) -> List[Dict[str, Any]]:
        """
        Run the MAP-Elites algorithm for multiple generations.
        
        Args:
            n_generations: Number of generations to run
            iterations_per_generation: Number of mutations per generation
            log_frequency: How often to print progress (in generations)
            save_frequency: How often to save the archive (in generations)
        
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
            
            # Update performance tracker
            self.performance_tracker.update(gen + 1, self.archive)
            
            # Save archive and database periodically
            if (gen + 1) % save_frequency == 0:
                self.save_archive(gen + 1)
                self.save_molecule_database()
            
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
        
        # Save final results
        self.save_archive(n_generations)
        self.save_molecule_database()
        self.performance_tracker.save()
        self.performance_plotter.plot_convergence(self.performance_tracker)
        self.performance_plotter.plot_hypervolume(self.performance_tracker)
        self.performance_plotter.plot_archive_heatmap(self.archive)
        self.performance_plotter.plot_archive_cells_evolution(self.performance_tracker)
        self.performance_plotter.plot_qd_score_evolution(self.performance_tracker)
        
        return history
    
    def save_archive(self, generation: int):
        """Save current archive to JSON"""
        archive_data = {
            'generation': generation,
            'solutions': []
        }
        
        for entry in self.archive.get_all_solutions():
            archive_data['solutions'].append({
                'indices': entry['indices'],
                'smiles': entry['solution'],
                'properties': entry['properties'],
                'objective': entry['objective']
            })
        
        filename = self.output_dir / f'archive_gen_{generation:04d}.json'
        import json
        with open(filename, 'w') as f:
            json.dump(archive_data, f, indent=2)
        
        print(f"Saved archive to {filename}")
    
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