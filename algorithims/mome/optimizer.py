import os
import sys
import random
import numpy as np
from typing import Callable, Dict, Any, Optional, List
from archive import MOMEArchive
from performance import MOMEPerformanceTracker
from plotting import MOMEPlotter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'molev_utils'))
from diversity_metrics import compute_diversity_metrics


class MOMEOptimizer:
    """
    Multi-Objective MAP-Elites (MOME) optimizer.
    Evolves Pareto fronts in each cell of the descriptor space.
    """
    
    def __init__(
        self,
        archive: MOMEArchive,
        generate_fn: Callable[[], Any],
        mutate_fn: Callable[[Any], Any],
        evaluate_fn: Callable[[Any], Dict[str, Any]],
        random_init_size: int = 100,
        output_dir: str = "mome_results",
        reference_point: List[float] = None,
        crossover_rate: float = 0.0,
        crossover_fn: Optional[Callable[[Any, Any], Any]] = None
    ):
        """
        Initialize the MOME optimizer.
        
        Args:
            archive: MOMEArchive instance for storing Pareto fronts
            generate_fn: Function that generates a new random solution
            mutate_fn: Function that mutates an existing solution
            evaluate_fn: Function that evaluates a solution and returns properties
            random_init_size: Number of random solutions for initialization
            output_dir: Directory for saving results
            reference_point: Reference point for hypervolume calculation
        """
        self.archive = archive
        self.generate_fn = generate_fn
        self.mutate_fn = mutate_fn
        self.evaluate_fn = evaluate_fn
        self.random_init_size = random_init_size
        self.crossover_rate = crossover_rate
        self.crossover_fn = crossover_fn
        
        from pathlib import Path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set reference point for hypervolume
        if reference_point is None:
            # Default reference point (should be set based on problem)
            reference_point = [0.0] * archive.n_objectives
        self.reference_point = np.array(reference_point)
        
        self.performance_tracker = MOMEPerformanceTracker(
            self.output_dir, 
            reference_point=reference_point
        )
        self.performance_plotter = MOMEPlotter(self.output_dir)

        # Statistics
        self.total_evaluations = 0
        self.generation = 0

        # Database of all evaluated molecules (like NSGA-II)
        self.all_molecules = []

        # Cache: SMILES -> properties dict (avoid redundant QC evaluations)
        self._eval_cache: Dict[str, Dict[str, Any]] = {}
    
    def _evaluate_with_cache(self, solution: str) -> Dict[str, Any]:
        """Evaluate a molecule, using cached result if already seen."""
        if solution in self._eval_cache:
            return self._eval_cache[solution]
        properties = self.evaluate_fn(solution)
        self.total_evaluations += 1
        if properties is not None:
            self._eval_cache[solution] = properties
        return properties

    def initialize(self) -> None:
        """Initialize the archive with random solutions."""
        print(f"Initializing MOME with {self.random_init_size} random solutions...")

        for i in range(self.random_init_size):
            solution = self.generate_fn()
            if solution is None:
                continue

            properties = self._evaluate_with_cache(solution)
            if properties is None:
                continue

            # Update molecule database (like NSGA-II)
            self.update_molecule_database(solution, properties, generation=0)

            self.archive.add(solution, properties)

            if (i + 1) % max(1, self.random_init_size // 10) == 0:
                print(f"  {i + 1}/{self.random_init_size} - "
                      f"Coverage: {self.archive.get_coverage():.2%}, "
                      f"Total solutions: {len(self.archive)}")

        print(f"Initialization complete.")
        print(f"  Filled cells: {self.archive.n_filled}")
        print(f"  Total solutions: {len(self.archive)}")
        print(f"  Coverage: {self.archive.get_coverage():.2%}")
    
    def step(self, n_iterations: int = 1) -> Dict[str, Any]:
        """
        Run iterations of MOME algorithm.
        
        Each iteration:
        1. Sample a cell uniformly from archive
        2. Sample a solution from that cell's Pareto front
        3. Mutate the solution
        4. Evaluate and add to archive
        
        Args:
            n_iterations: Number of iterations to run
        
        Returns:
            Dictionary with statistics
        """
        n_added = 0
        
        for _ in range(n_iterations):
            # Sample a parent from the archive
            parent = self.archive.sample_solution_from_archive()
            
            if parent is None:
                # Archive is empty, generate random solution
                solution = self.generate_fn()
            elif (self.crossover_fn is not None
                  and self.crossover_rate > 0.0
                  and random.random() < self.crossover_rate):
                # Crossover: sample a second parent and recombine
                parent2 = self.archive.sample_solution_from_archive()
                if parent2 is not None:
                    solution = self.crossover_fn(parent, parent2)
                else:
                    solution = self.mutate_fn(parent)
            else:
                # Mutate the parent
                solution = self.mutate_fn(parent)
            
            if solution is None:
                continue
            
            # Evaluate the solution (uses cache if already seen)
            properties = self._evaluate_with_cache(solution)
            if properties is None:
                continue

            # Update molecule database (like NSGA-II)
            self.update_molecule_database(solution, properties, generation=self.generation + 1)

            # Try to add to archive
            was_added = self.archive.add(solution, properties)
            
            if was_added:
                n_added += 1
        
        self.generation += 1
        
        # Calculate MOQD score using PyMoo (same as NSGA-II)
        # MOQD should never decrease because we only add non-dominated solutions
        moqd_score = self.archive.compute_moqd_score()

        # Calculate global hypervolume using PyMoo (same as NSGA-II)
        global_hv = self.archive.compute_global_hypervolume()
        
        # Diversity metrics over all archive members
        archive_smiles = [
            entry['solution']
            for _, front in self.archive.iter_filled_cells()
            for entry in front
        ]
        div = compute_diversity_metrics(archive_smiles, max_sample=500)

        return {
            'generation': self.generation,
            'n_added': n_added,
            'archive_size': len(self.archive),
            'filled_cells': self.archive.n_filled,
            'coverage': self.archive.get_coverage(),
            'moqd_score': moqd_score,
            'global_hypervolume': global_hv,
            'total_evaluations': self.total_evaluations,
            'int_div': div['int_div'],
            'scaffold_count': div['scaffold_count'],
            'n_unique': div['n_unique'],
        }

    def update_molecule_database(self, solution: Any, properties: Dict[str, Any], generation: int):
        """
        Update the molecule database with a new evaluated molecule (like NSGA-II).

        Args:
            solution: SMILES string
            properties: Dictionary with all properties including objectives and measures
            generation: Generation number
        """
        # Check if molecule already exists
        existing = next((m for m in self.all_molecules if m['smiles'] == solution), None)

        if existing:
            # Update if this is from an earlier generation
            if generation < existing['generation']:
                existing['generation'] = generation
                existing['objectives'] = properties.get('objectives', [])

                # Update all properties
                for key, value in properties.items():
                    if key != 'smiles':
                        existing[key] = value
        else:
            # Create new entry
            mol_entry = {
                'smiles': solution,
                'generation': generation,
                'objectives': properties.get('objectives', [])
            }

            # Add all objective values as named fields
            for i, obj_key in enumerate(self.archive.objective_keys):
                if obj_key in properties:
                    mol_entry[obj_key] = properties[obj_key]

            # Add all other properties
            for key, value in properties.items():
                if key not in mol_entry:
                    mol_entry[key] = value

            self.all_molecules.append(mol_entry)

    def save_molecule_database(self):
        """
        Save the complete molecule database to JSON file (like NSGA-II).

        Saves all molecules with their objectives and properties to
        'all_molecules_database.json' in the output directory.
        """
        db_file = self.output_dir / "all_molecules_database.json"
        import json

        # Sort by generation for easier analysis
        sorted_molecules = sorted(self.all_molecules, key=lambda x: x.get('generation', 0))

        with open(db_file, 'w') as f:
            json.dump(sorted_molecules, f, indent=2)

        print(f"Saved {len(sorted_molecules)} molecules to {db_file}")

    def run(self, n_generations: int, iterations_per_generation: int = 1,
            log_frequency: int = 10, save_frequency: int = 50) -> List[Dict[str, Any]]:
        """
        Run MOME for multiple generations.
        
        Args:
            n_generations: Number of generations
            iterations_per_generation: Mutations per generation
            log_frequency: How often to log (in generations)
            save_frequency: How often to save archive (in generations)
        
        Returns:
            List of statistics per generation
        """
        if len(self.archive) == 0:
            print("Archive is empty. Running initialization...")
            self.initialize()
        
        print(f"\nRunning MOME for {n_generations} generations "
              f"({iterations_per_generation} iterations/gen)...")
        
        history = []
        
        for gen in range(n_generations):
            stats = self.step(iterations_per_generation)
            
            # Update performance tracker (adapted for MOME)
            self.performance_tracker.update(gen + 1, self.archive)
            
            # Save archive and database periodically
            if (gen + 1) % save_frequency == 0:
                self.save_archive(gen + 1)
                self.save_molecule_database()

            history.append(stats)
            
            if (gen + 1) % log_frequency == 0:
                print(f"Gen {stats['generation']:4d}: "
                      f"Coverage={stats['coverage']:6.2%}, "
                      f"Solutions={stats['archive_size']:5d}, "
                      f"Cells={stats['filled_cells']:4d}, "
                      f"MOQD={stats['moqd_score']:8.2f}, "
                      f"GlobalHV={stats['global_hypervolume']:8.2f}, "
                      f"Added={stats['n_added']:3d}/{iterations_per_generation}")
        
        print("\nOptimization complete!")
        print(f"Final coverage: {self.archive.get_coverage():.2%}")
        print(f"Final filled cells: {self.archive.n_filled}")
        print(f"Final total solutions: {len(self.archive)}")
        print(f"Total evaluations: {self.total_evaluations}")
        
        # Save final results
        self.save_archive(n_generations)
        self.save_molecule_database()
        self.performance_tracker.save()
        
        # # Generate plots  # Commented out to avoid calling missing methods in plotting.py
        # self.performance_plotter.plot_convergence(self.performance_tracker)
        # self.performance_plotter.plot_hypervolume(self.performance_tracker)
        # self.performance_plotter.plot_archive_heatmap_mome(self.archive, self.reference_point)
        # self.performance_plotter.plot_archive_cells_evolution(self.performance_tracker)
        # self.performance_plotter.plot_qd_score_evolution(self.performance_tracker)
        
        return history
    
    def save_archive(self, generation: int):
        """Save current archive to JSON."""
        archive_data = {
            'generation': generation,
            'n_objectives': self.archive.n_objectives,
            'objective_keys': self.archive.objective_keys,
            'cells': []
        }
        
        # Save each cell's Pareto front
        for idx, front in self.archive.iter_filled_cells():
            cell_data = {
                'indices': idx,
                'front_size': len(front),
                'solutions': []
            }

            for entry in front:
                cell_data['solutions'].append({
                    'smiles': entry['solution'],
                    'properties': entry['properties'],
                    'objectives': entry['objectives'].tolist()
                })

            archive_data['cells'].append(cell_data)
        
        filename = self.output_dir / f'mome_archive_gen_{generation:04d}.json'
        import json
        with open(filename, 'w') as f:
            json.dump(archive_data, f, indent=2)
        
        print(f"Saved MOME archive to {filename}")
    
    def get_best_solutions(self, n: int = 5) -> List[Dict[str, Any]]:
        """
        Get top n solutions from the global Pareto front.
        
        Returns:
            List of solution dictionaries
        """
        global_front = self.archive.get_global_pareto_front()
        
        if not global_front:
            return []
        
        # Sort by sum of objectives (simple approach)
        sorted_front = sorted(
            global_front,
            key=lambda x: np.sum(x['objectives']),
            reverse=True
        )
        
        return sorted_front[:n]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the archive."""
        global_front = self.archive.get_global_pareto_front()

        return {
            'filled_cells': self.archive.n_filled,
            'total_solutions': len(self.archive),
            'coverage': self.archive.get_coverage(),
            'moqd_score': self.archive.compute_moqd_score(),
            'global_front_size': len(global_front),
            'global_hypervolume': self.archive.compute_global_hypervolume(),
            'total_evaluations': self.total_evaluations,
            'generations': self.generation
        }

    @staticmethod
    def recalculate_from_database(results_dir: str, archive_config: Dict = None):
        """
        Recalculate archive, hypervolume, MOQD, and plots from existing all_molecules_database.json.
        Similar to NSGA-II's recalculate_from_database function.

        Args:
            results_dir: Path to the results directory containing all_molecules_database.json
            archive_config: Dictionary with archive configuration (measure_dims, measure_keys, objective_keys, etc.)
        """
        import json
        from pathlib import Path

        results_path = Path(results_dir)
        db_file = results_path / 'all_molecules_database.json'

        if not db_file.exists():
            print(f"ERROR: Database file not found: {db_file}")
            return

        # Load database
        with open(db_file, 'r') as f:
            db = json.load(f)

        print(f"Loaded {len(db)} molecules from {db_file}")

        # Find max generation
        max_gen = max((m.get('generation', 0) for m in db), default=0)
        print(f"Max generation: {max_gen}")

        # Get archive configuration from first molecule or use defaults
        if archive_config is None:
            # Try to infer from database
            sample_mol = db[0] if db else {}
            objective_keys = []

            # Common objective names
            for key in ['beta_mean', 'homo_lumo_gap', 'dipole_moment', 'alpha_mean', 'gamma', 'total_energy']:
                if key in sample_mol:
                    objective_keys.append(key)

            if not objective_keys and 'objectives' in sample_mol:
                # Use generic names
                objective_keys = [f'obj_{i}' for i in range(len(sample_mol['objectives']))]

            archive_config = {
                'measure_dims': [10, 10],
                'measure_keys': ['num_atoms_bin', 'num_bonds_bin'],
                'objective_keys': objective_keys,
                'max_front_size': 50,
                'optimize_objectives': [('max', None)] * len(objective_keys)
            }

            print(f"Inferred archive config: {objective_keys}")

        # Reconstruct metrics for each generation
        metrics = {
            'generation': [],
            'hypervolume': [],
            'moqd': [],
            'coverage': [],
            'filled_cells': [],
            'total_solutions': [],
            'global_front_size': []
        }

        for gen in range(max_gen + 1):
            # Get molecules up to this generation
            molecules_up_to_gen = [m for m in db if m.get('generation', 0) <= gen]
            if not molecules_up_to_gen:
                continue

            # Create archive
            archive = MOMEArchive(**archive_config)

            # Rebuild archive
            for mol in molecules_up_to_gen:
                if 'objectives' in mol and 'smiles' in mol:
                    # Add to archive
                    archive.add(mol['smiles'], mol)

            # Compute metrics
            global_hv = archive.compute_global_hypervolume()
            moqd = archive.compute_moqd_score()

            # Append to metrics
            metrics['generation'].append(gen)
            metrics['hypervolume'].append(global_hv)
            metrics['moqd'].append(moqd)
            metrics['coverage'].append(archive.get_coverage())
            metrics['filled_cells'].append(archive.n_filled)
            metrics['total_solutions'].append(len(archive))
            metrics['global_front_size'].append(len(archive.get_global_pareto_front()))

            print(f"Generation {gen}: HV={global_hv:.6f}, MOQD={moqd:.6f}, "
                  f"Coverage={archive.get_coverage():.2%}, Cells={archive.n_filled}")

        # Save reconstructed metrics
        metrics_file = results_path / 'mome_performance_metrics_recalculated.json'
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nSaved reconstructed performance metrics to {metrics_file}")

        # Create final archive from all molecules
        final_archive = MOMEArchive(**archive_config)
        for mol in db:
            if 'objectives' in mol and 'smiles' in mol:
                final_archive.add(mol['smiles'], mol)

        print(f"\nFinal archive rebuilt with {final_archive.total_individuals()} individuals "
              f"in {final_archive.n_filled} cells")
        print(f"Coverage: {final_archive.get_coverage():.2%}")
        print(f"Global HV: {final_archive.compute_global_hypervolume():.6f}")
        print(f"MOQD Score: {final_archive.compute_moqd_score():.6f}")

        # Save the rebuilt archive
        archive_data = {
            'generation': max_gen,
            'n_objectives': final_archive.n_objectives,
            'objective_keys': final_archive.objective_keys,
            'cells': []
        }

        for idx, front in final_archive.iter_filled_cells():
            cell_data = {
                'indices': idx,
                'front_size': len(front),
                'solutions': []
            }

            for entry in front:
                cell_data['solutions'].append({
                    'smiles': entry['solution'],
                    'properties': entry['properties'],
                    'objectives': entry['objectives'].tolist()
                })

            archive_data['cells'].append(cell_data)

        archive_file = results_path / 'mome_archive_recalculated.json'
        with open(archive_file, 'w') as f:
            json.dump(archive_data, f, indent=2)
        print(f"Saved rebuilt archive to {archive_file}")

        print("\nRecalculation complete!")