import numpy as np
from typing import List, Tuple, Optional, Any, Dict
from pymoo.indicators.hv import Hypervolume

"""
Multi-Objective MAP-Elites Archive with Improved Pareto Front Management
Uses crowding distance instead of random removal for better hypervolume preservation
Uses PyMoo for hypervolume calculations (same as NSGA-II implementation)
"""

class MOMEArchive:
    """
    Archive for Multi-Objective Quality Diversity optimization using MOME algorithm.
    Stores Pareto fronts in a multidimensional grid based on behavioral measures.
    """

    def __init__(self, measure_dims: List[int], measure_keys: List[str],
                 objective_keys: List[str], max_front_size: int = 50,
                 use_crowding_distance: bool = True,
                 optimize_objectives: List[Tuple[str, Any]] = None):
        """
        Initialize the MOME archive.

        Args:
            measure_dims: List of integers specifying the number of bins along each dimension.
            measure_keys: List of dictionary keys that define the measures (in order).
            objective_keys: List of dictionary keys that define the objectives.
            max_front_size: Maximum number of solutions in each Pareto front
            use_crowding_distance: If True, use crowding distance for removal; if False, use random
            optimize_objectives: List of tuples (opt_type, target) where opt_type is 'max', 'min', or 'target'
                               If None, defaults to maximization for all objectives
        """
        self.measure_dims = tuple(measure_dims)
        self.n_dims = len(measure_dims)
        self.measure_keys = measure_keys
        self.objective_keys = objective_keys
        self.n_objectives = len(objective_keys)
        self.max_front_size = max_front_size
        self.use_crowding_distance = use_crowding_distance

        # Set optimization objectives (default to maximization)
        if optimize_objectives is None:
            self.optimize_objectives = [('max', None)] * self.n_objectives
        else:
            self.optimize_objectives = optimize_objectives

        if len(measure_keys) != self.n_dims:
            raise ValueError(f"Number of measure_keys ({len(measure_keys)}) must match "
                           f"number of dimensions ({self.n_dims})")

        # Create grids for storing Pareto fronts
        self.fronts = np.empty(self.measure_dims, dtype=object)
        for idx in np.ndindex(self.measure_dims):
            self.fronts[idx] = []

        # Track statistics
        self.n_filled = 0
        self.n_updates = 0

        # Initialize PyMoo hypervolume indicator (same as NSGA-II)
        self.hv_indicator = Hypervolume(ref_point=self._get_ref_point())

    def _get_ref_point(self):
        """Get reference point for hypervolume calculation in transformed (minimization) space."""
        ref = []
        for opt, _ in self.optimize_objectives:
            if opt == 'max':
                # For maximization (transformed to -obj), reference is 0.0 (worse than negative values)
                ref.append(0.0)
            else:
                # For minimization, reference is a large value
                ref.append(1000.0)
        return np.array(ref)

    def _transform_objectives(self, objectives: np.ndarray) -> np.ndarray:
        """Transform objectives for minimization (negate maximization objectives)."""
        transformed = objectives.copy()
        for i, (opt, _) in enumerate(self.optimize_objectives):
            if opt == 'max':
                transformed[:, i] = -transformed[:, i]
        return transformed

    def _extract_measures(self, properties: Dict[str, Any]) -> Optional[Tuple[int, ...]]:
        """Extract measure values and convert to grid indices."""
        try:
            measures = [properties[key] for key in self.measure_keys]
        except KeyError as e:
            raise ValueError(f"Missing required measure key: {e}")
        
        indices = []
        for measure, dim_size in zip(measures, self.measure_dims):
            idx = int(measure)
            if idx < 0 or idx >= dim_size:
                return None
            indices.append(idx)
        
        return tuple(indices)
    
    def _extract_objectives(self, properties: Dict[str, Any]) -> np.ndarray:
        """Extract objective values as array."""
        try:
            return np.array([float(properties[key]) for key in self.objective_keys])
        except KeyError as e:
            raise ValueError(f"Missing required objective key: {e}")
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid objective value: {e}")
    
    def _dominates(self, obj1: np.ndarray, obj2: np.ndarray) -> bool:
        """Check if obj1 strictly dominates obj2 (all >= and at least one >)."""
        return np.all(obj1 >= obj2) and np.any(obj1 > obj2)
    
    def _calculate_crowding_distance(self, front: List[Dict]) -> np.ndarray:
        """
        Calculate crowding distance for each solution in the front.
        Solutions with larger crowding distance are more isolated (less crowded).
        
        Returns:
            Array of crowding distances, one per solution
        """
        n = len(front)
        if n <= 2:
            return np.full(n, np.inf)
        
        # Extract objectives
        objectives = np.array([entry['objectives'] for entry in front])
        
        # Initialize distances
        distances = np.zeros(n)
        
        # Calculate crowding distance for each objective
        for m in range(self.n_objectives):
            # Sort by m-th objective
            sorted_indices = np.argsort(objectives[:, m])
            
            # Boundary solutions get infinite distance
            distances[sorted_indices[0]] = np.inf
            distances[sorted_indices[-1]] = np.inf
            
            # Calculate distances for interior solutions
            obj_range = objectives[sorted_indices[-1], m] - objectives[sorted_indices[0], m]
            
            if obj_range > 0:
                for i in range(1, n - 1):
                    distances[sorted_indices[i]] += (
                        objectives[sorted_indices[i + 1], m] - 
                        objectives[sorted_indices[i - 1], m]
                    ) / obj_range
        
        return distances
    
    def _update_pareto_front(self, front: List[Dict], new_solution: Any, 
                           new_properties: Dict[str, Any], new_objectives: np.ndarray) -> Tuple[List[Dict], bool]:
        """
        Update a Pareto front with a new solution.
        
        Returns:
            Updated front and whether the solution was added
        """
        # Check if new solution is dominated by any solution in front
        for entry in front:
            if self._dominates(entry['objectives'], new_objectives):
                return front, False  # New solution is dominated, reject it
        
        # New solution is not dominated, add it and remove dominated solutions
        new_front = []
        for entry in front:
            if not self._dominates(new_objectives, entry['objectives']):
                new_front.append(entry)
        
        # Add new solution
        new_front.append({
            'solution': new_solution,
            'properties': new_properties.copy(),
            'objectives': new_objectives
        })
        
        # If front exceeds max size, remove solution
        if len(new_front) > self.max_front_size:
            if self.use_crowding_distance:
                # Remove solution with smallest crowding distance (most crowded)
                distances = self._calculate_crowding_distance(new_front)
                # Don't remove boundary points (infinite distance)
                finite_indices = np.where(np.isfinite(distances))[0]
                if len(finite_indices) > 0:
                    remove_idx = finite_indices[np.argmin(distances[finite_indices])]
                else:
                    # All have infinite distance, remove randomly
                    remove_idx = np.random.randint(len(new_front))
            else:
                # Random removal (original MOME paper)
                remove_idx = np.random.randint(len(new_front))
            
            new_front.pop(remove_idx)
        
        return new_front, True
    
    def add(self, solution: Any, properties: Dict[str, Any]) -> bool:
        """
        Add a solution to the archive if it's not dominated in its cell.
        
        Args:
            solution: The solution object to store
            properties: Dictionary containing all properties including objectives and measures
        
        Returns:
            True if the solution was added, False otherwise
        """
        # Extract objectives and measures
        objectives = self._extract_objectives(properties)
        idx = self._extract_measures(properties)
        
        # Out of bounds
        if idx is None:
            return False
        
        # Get current front
        current_front = self.fronts[idx]
        was_empty = len(current_front) == 0
        
        # Update Pareto front
        new_front, was_added = self._update_pareto_front(
            current_front, solution, properties, objectives
        )
        
        if was_added:
            self.fronts[idx] = new_front
            
            # Update statistics
            if was_empty and len(new_front) > 0:
                self.n_filled += 1
            elif not was_empty and len(new_front) == 0:
                self.n_filled -= 1
            
            self.n_updates += 1
            return True
        
        return False
    
    def get_front(self, measures: List[float]) -> Optional[List[Dict[str, Any]]]:
        """
        Retrieve the Pareto front at the given measures.
        
        Returns:
            List of solution entries if cell is filled, None otherwise
        """
        if len(measures) != self.n_dims:
            raise ValueError(f"Expected {self.n_dims} measures, got {len(measures)}")
        
        indices = []
        for measure, dim_size in zip(measures, self.measure_dims):
            idx = int(measure)
            if idx < 0 or idx >= dim_size:
                return None
            indices.append(idx)
        
        idx = tuple(indices)
        front = self.fronts[idx]
        
        return front if len(front) > 0 else None
    
    def sample_solution_from_archive(self) -> Optional[Any]:
        """
        Sample a solution uniformly from the archive.
        First sample a cell, then sample a solution from its Pareto front.
        """
        # Get all filled cells
        filled_cells = []
        for idx in np.ndindex(self.measure_dims):
            if len(self.fronts[idx]) > 0:
                filled_cells.append(idx)
        
        if not filled_cells:
            return None
        
        # Sample a random cell
        cell_idx = filled_cells[np.random.randint(len(filled_cells))]
        
        # Sample a random solution from the front
        front = self.fronts[cell_idx]
        solution_idx = np.random.randint(len(front))
        
        return front[solution_idx]['solution']
    
    def get_all_solutions(self) -> List[Dict[str, Any]]:
        """Get all solutions in the archive."""
        results = []
        
        for idx in np.ndindex(self.measure_dims):
            front = self.fronts[idx]
            for entry in front:
                results.append({
                    'indices': idx,
                    'solution': entry['solution'],
                    'properties': entry['properties'],
                    'objectives': entry['objectives']
                })
        
        return results
    
    def get_coverage(self) -> float:
        """Get the proportion of filled cells."""
        total_cells = np.prod(self.measure_dims)
        return self.n_filled / total_cells
    
    def _calculate_cell_hypervolume(self, front: List[Dict]) -> float:
        """
        Calculate hypervolume for a single cell's Pareto front using PyMoo.
        Uses the same approach as NSGA-II BinnedParetoArchive.
        """
        if not front:
            return 0.0

        # Extract objectives and transform for minimization
        objectives = np.array([entry['objectives'] for entry in front])
        transformed_objectives = self._transform_objectives(objectives)

        # Calculate hypervolume using PyMoo
        try:
            return float(self.hv_indicator(transformed_objectives))
        except Exception:
            return 0.0

    def get_hypervolume_per_cell(self) -> np.ndarray:
        """
        Calculate hypervolume for each cell's Pareto front using PyMoo.
        This is used for MOQD score calculation.
        """
        hypervolumes = np.zeros(self.measure_dims)

        for idx in np.ndindex(self.measure_dims):
            front = self.fronts[idx]
            if len(front) > 0:
                hypervolumes[idx] = self._calculate_cell_hypervolume(front)

        return hypervolumes

    def compute_moqd_score(self) -> float:
        """
        Compute MOQD score as sum of hypervolumes of individual cell Pareto fronts.
        Same as NSGA-II BinnedParetoArchive.compute_moqd_score().
        MOQD should never decrease because we only add non-dominated solutions.
        """
        total_score = 0.0
        for idx in np.ndindex(self.measure_dims):
            front = self.fronts[idx]
            if front:
                total_score += self._calculate_cell_hypervolume(front)
        return total_score

    def get_moqd_score(self) -> float:
        """Alias for compute_moqd_score() for backward compatibility."""
        return self.compute_moqd_score()

    def compute_global_hypervolume(self) -> float:
        """
        Compute global hypervolume by combining all Pareto fronts and calculating HV on the overall front.
        Same as NSGA-II BinnedParetoArchive.compute_global_hypervolume().
        """
        # Get global Pareto front
        global_front = self.get_global_pareto_front()

        if not global_front:
            return 0.0

        # Extract objectives and transform
        objectives = np.array([entry['objectives'] for entry in global_front])
        transformed_objectives = self._transform_objectives(objectives)

        # Calculate hypervolume using PyMoo
        try:
            return float(self.hv_indicator(transformed_objectives))
        except Exception:
            return 0.0
    
    def get_global_pareto_front(self) -> List[Dict[str, Any]]:
        """Get the global Pareto front across all cells."""
        all_solutions = self.get_all_solutions()
        
        if not all_solutions:
            return []
        
        global_front = []
        for candidate in all_solutions:
            is_dominated = False
            
            for entry in global_front:
                if self._dominates(entry['objectives'], candidate['objectives']):
                    is_dominated = True
                    break
            
            if not is_dominated:
                global_front = [entry for entry in global_front 
                              if not self._dominates(candidate['objectives'], entry['objectives'])]
                global_front.append(candidate)
        
        return global_front
    
    def __len__(self) -> int:
        """Return the total number of solutions in archive."""
        return sum(len(self.fronts[idx]) for idx in np.ndindex(self.measure_dims))
    
    def __repr__(self) -> str:
        removal_strategy = "crowding" if self.use_crowding_distance else "random"
        return (f"MOMEArchive(dims={self.measure_dims}, "
                f"measures={self.measure_keys}, objectives={self.objective_keys}, "
                f"filled={self.n_filled}/{np.prod(self.measure_dims)}, "
                f"total_solutions={len(self)}, "
                f"coverage={self.get_coverage():.2%}, "
                f"removal={removal_strategy})")