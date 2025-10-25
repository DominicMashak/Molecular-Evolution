import numpy as np
from typing import List, Tuple, Optional, Any, Dict

"""
Prototype by Claude 
"""

class MAPElitesArchive:
    """
    Archive for Quality Diversity optimization using MAP-Elites algorithm.
    Stores solutions in a multidimensional grid based on behavioral measures.
    """
    
    def __init__(self, measure_dims: List[int], measure_keys: List[str], objective_key: str):
        """
        Initialize the MAP-Elites archive.
        
        Args:
            measure_dims: List of integers specifying the number of bins along each dimension.
                         e.g., [10, 20, 15] creates a 10x20x15 grid
            measure_keys: List of dictionary keys that define the measures (in order).
                         e.g., ['speed', 'height'] means first dimension is 'speed', second is 'height'
            objective_key: Dictionary key that defines the objective to maximize.
                          e.g., 'fitness'
        """
        self.measure_dims = tuple(measure_dims)
        self.n_dims = len(measure_dims)
        self.measure_keys = measure_keys
        self.objective_key = objective_key
        
        if len(measure_keys) != self.n_dims:
            raise ValueError(f"Number of measure_keys ({len(measure_keys)}) must match "
                           f"number of dimensions ({self.n_dims})")
        
        # Create grids for storing objectives, solutions, and all properties
        self.objectives = np.full(self.measure_dims, -np.inf, dtype=np.float64)
        self.solutions = np.empty(self.measure_dims, dtype=object)
        self.properties = np.empty(self.measure_dims, dtype=object)
        
        # Track statistics
        self.n_filled = 0
        self.n_updates = 0
    
    def _extract_measures(self, properties: Dict[str, Any]) -> Optional[Tuple[int, ...]]:
        """
        Extract measure values from properties dictionary and convert to grid indices.
        
        Args:
            properties: Dictionary containing solution properties
        
        Returns:
            Tuple of indices, or None if measures are out of bounds or missing
        """
        try:
            measures = [properties[key] for key in self.measure_keys]
        except KeyError as e:
            raise ValueError(f"Missing required measure key: {e}")
        
        indices = []
        for measure, dim_size in zip(measures, self.measure_dims):
            # Convert to integer index
            idx = int(measure)
            if idx < 0 or idx >= dim_size:
                return None
            indices.append(idx)
        
        return tuple(indices)
    
    def _extract_objective(self, properties: Dict[str, Any]) -> float:
        """
        Extract objective value from properties dictionary.
        
        Args:
            properties: Dictionary containing solution properties
        
        Returns:
            Objective value
        """
        try:
            return float(properties[self.objective_key])
        except KeyError:
            raise ValueError(f"Missing required objective key: '{self.objective_key}'")
        except (ValueError, TypeError):
            raise ValueError(f"Invalid objective value for '{self.objective_key}': {properties.get(self.objective_key)}")
    
    def add(self, solution: Any, properties: Dict[str, Any]) -> bool:
        """
        Add a solution to the archive if it improves the corresponding cell.
        
        Args:
            solution: The solution object to store (can be any type)
            properties: Dictionary containing all properties of the solution,
                       including the objective and measures
        
        Returns:
            True if the solution was added (new cell or improvement), False otherwise
        """
        # Extract objective and measures from properties
        objective = self._extract_objective(properties)
        idx = self._extract_measures(properties)
        
        # Out of bounds
        if idx is None:
            return False
        
        # Check if this improves the cell
        current_obj = self.objectives[idx]
        
        if objective > current_obj:
            # Track if this is a new cell
            was_empty = np.isinf(current_obj)
            
            # Store the solution and all properties
            self.objectives[idx] = objective
            self.solutions[idx] = solution
            self.properties[idx] = properties.copy()  # Store a copy to avoid external mutations
            
            # Update statistics
            if was_empty:
                self.n_filled += 1
            self.n_updates += 1
            
            return True
        
        return False
    
    def get(self, measures: List[float]) -> Optional[Dict[str, Any]]:
        """
        Retrieve the solution and properties at the given measures.
        
        Args:
            measures: List of measure values (bin indices)
        
        Returns:
            Dictionary with 'solution', 'properties', and 'objective' keys if cell is filled,
            None otherwise
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
        
        if np.isinf(self.objectives[idx]):
            return None
        
        return {
            'solution': self.solutions[idx],
            'properties': self.properties[idx],
            'objective': self.objectives[idx]
        }
    
    def get_all_solutions(self) -> List[Dict[str, Any]]:
        """
        Get all solutions in the archive.
        
        Returns:
            List of dictionaries, each containing:
                - 'indices': tuple of grid indices
                - 'solution': the stored solution
                - 'properties': all properties (including objective and measures)
                - 'objective': the objective value (for convenience)
        """
        results = []
        
        # Iterate over all indices
        for idx in np.ndindex(self.measure_dims):
            if not np.isinf(self.objectives[idx]):
                results.append({
                    'indices': idx,
                    'solution': self.solutions[idx],
                    'properties': self.properties[idx],
                    'objective': self.objectives[idx]
                })
        
        return results
    
    def get_coverage(self) -> float:
        """
        Get the proportion of filled cells in the archive.
        
        Returns:
            Coverage as a fraction between 0 and 1
        """
        total_cells = np.prod(self.measure_dims)
        return self.n_filled / total_cells
    
    def get_max_objective(self) -> float:
        """
        Get the maximum objective value in the archive.
        
        Returns:
            Maximum objective value, or -inf if archive is empty
        """
        if self.n_filled == 0:
            return -np.inf
        return np.max(self.objectives[~np.isinf(self.objectives)])
    
    def get_mean_objective(self) -> float:
        """
        Get the mean objective value of filled cells.
        
        Returns:
            Mean objective value, or -inf if archive is empty
        """
        if self.n_filled == 0:
            return -np.inf
        return np.mean(self.objectives[~np.isinf(self.objectives)])
    
    def __len__(self) -> int:
        """Return the number of filled cells."""
        return self.n_filled
    
    def __repr__(self) -> str:
        return (f"MAPElitesArchive(dims={self.measure_dims}, "
                f"measures={self.measure_keys}, objective='{self.objective_key}', "
                f"filled={self.n_filled}/{np.prod(self.measure_dims)}, "
                f"coverage={self.get_coverage():.2%})")


# Example usage
if __name__ == "__main__":
    # Create a 10x10 archive with 'speed' and 'height' as measures, 'fitness' as objective
    archive = MAPElitesArchive(
        measure_dims=[10, 10],
        measure_keys=['speed', 'height'],
        objective_key='fitness'
    )
    
    # Add some solutions with various properties
    archive.add(
        solution="genome_1",
        properties={
            'fitness': 5.0,
            'speed': 2,
            'height': 3,
            'energy_cost': 12.5,
            'generation': 0
        }
    )
    
    archive.add(
        solution="genome_2",
        properties={
            'fitness': 3.0,  # Lower fitness, won't be added
            'speed': 2,
            'height': 3,
            'energy_cost': 10.0,
            'generation': 1
        }
    )
    
    archive.add(
        solution="genome_3",
        properties={
            'fitness': 8.0,
            'speed': 5,
            'height': 7,
            'energy_cost': 20.3,
            'generation': 1
        }
    )
    
    archive.add(
        solution="genome_4",
        properties={
            'fitness': 7.0,  # Higher fitness, will replace genome_1
            'speed': 2,
            'height': 3,
            'energy_cost': 15.0,
            'generation': 2
        }
    )
    
    print(archive)
    print(f"Max objective: {archive.get_max_objective()}")
    print(f"Mean objective: {archive.get_mean_objective()}")
    
    # Retrieve a specific solution
    result = archive.get([2, 3])
    if result:
        print(f"\nSolution at [2, 3]:")
        print(f"  Solution: {result['solution']}")
        print(f"  Objective: {result['objective']}")
        print(f"  All properties: {result['properties']}")
    
    # Get all solutions
    print(f"\nAll solutions: {len(archive.get_all_solutions())}")
    for entry in archive.get_all_solutions():
        print(f"  {entry['indices']}: {entry['solution']}")
        print(f"    Fitness: {entry['properties']['fitness']}, "
              f"Energy: {entry['properties']['energy_cost']}, "
              f"Gen: {entry['properties']['generation']}")