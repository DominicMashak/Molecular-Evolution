from typing import Dict, List, Tuple
from collections import defaultdict
import numpy as np
from individual import Individual
from dominance import dominates

class BinnedParetoArchive:
    """
    BinnedParetoArchive: A binned archive for NSGA-II, maintaining Pareto fronts per bin based on features.
    This class implements a binned archive that discretizes individuals into bins based on their features
    and maintains a Pareto front within each bin. It is designed for use in multi-objective optimization
    algorithms like NSGA-II, where diversity is promoted by binning the feature space.
    Attributes:
        n_bins (int): Number of bins per feature dimension for discretization.
        max_size (int): Maximum total number of individuals allowed in the archive.
        cells (Dict[Tuple[int, ...], List[Individual]]): Dictionary mapping bin coordinates to lists of individuals in that bin.
        feature_bounds (List[Tuple[float, float]]): List of (min, max) bounds for each feature.
        bin_visits (defaultdict): Tracks the number of visits to each bin.
        optimize_objectives (List[Tuple[str, Any]]): List specifying optimization direction ('max' or 'min') for each objective.
    Methods:
        __init__(n_bins: int = 10, max_size: int = 1000, optimize_objectives=None):
            Initializes the archive with specified number of bins, max size, and optimization objectives.
        update_bounds(features: List[float]):
            Updates the minimum and maximum bounds for each feature based on the provided features.
        discretize(features: List[float]) -> Tuple[int, ...]:
            Discretizes the given features into bin coordinates based on the current feature bounds.
        add(individual: Individual) -> bool:
            Attempts to add an individual to the archive. Updates bounds, discretizes features, and adds to the bin's Pareto front if not dominated.
            Prunes the archive if it exceeds max_size. Returns True if added, False otherwise.
        _prune_to_size():
            Prunes the archive to max_size by removing individuals from least visited bins.
        get_all_individuals() -> List[Individual]:
            Returns a list of all individuals currently in the archive across all bins.
        total_individuals() -> int:
            Returns the total number of individuals in the archive.
        size() -> int:
            Returns the number of occupied bins in the archive.
    """
    """Binned archive for NSGA-II, maintaining Pareto fronts per bin based on features."""
    
    def __init__(self, n_bins: int = 10, max_size: int = 1000, optimize_objectives=None):
        self.n_bins = n_bins
        self.max_size = max_size
        self.cells: Dict[Tuple[int, ...], List[Individual]] = {}
        self.feature_bounds = [(float('inf'), float('-inf')) for _ in range(4)]  # For 4 features
        self.bin_visits = defaultdict(int)
        self.optimize_objectives = optimize_objectives or [('max', None)] * 2  # Default
    
    def update_bounds(self, features: List[float]):
        """Update bounds for each feature."""
        for i, value in enumerate(features):
            min_val, max_val = self.feature_bounds[i]
            self.feature_bounds[i] = (min(min_val, value), max(max_val, value))
    
    def discretize(self, features: List[float]) -> Tuple[int, ...]:
        """Discretize features into bins."""
        bins = []
        for i, value in enumerate(features):
            min_val, max_val = self.feature_bounds[i]
            if max_val == min_val:
                bin_idx = 0
            else:
                normalized = (value - min_val) / (max_val - min_val)
                bin_idx = int(np.clip(normalized * self.n_bins, 0, self.n_bins - 1))
            bins.append(bin_idx)
        return tuple(bins)
    
    def add(self, individual: Individual) -> bool:
        """Add individual to archive if it improves the bin's Pareto front."""
        features = [individual.homo_lumo_gap, individual.transition_dipole, 
                    individual.oscillator_strength, individual.gamma]
        self.update_bounds(features)
        bin_coords = self.discretize(features)
        self.bin_visits[bin_coords] += 1
        
        if bin_coords not in self.cells:
            self.cells[bin_coords] = [individual]
            return True
        
        current_front = self.cells[bin_coords]
        
        # Check if new individual is dominated
        is_dominated = any(dominates(existing, individual, self.optimize_objectives) for existing in current_front)
        if is_dominated:
            return False
        
        # Remove dominated individuals
        non_dominated = [existing for existing in current_front
                         if not dominates(individual, existing, self.optimize_objectives)]
        non_dominated.append(individual)
        
        self.cells[bin_coords] = non_dominated
        
        # Prune if total archive size exceeds max_size
        if self.total_individuals() > self.max_size:
            self._prune_to_size()
        
        return True
    
    def _prune_to_size(self):
        """Prune archive to max_size by removing least visited bins."""
        all_inds = self.get_all_individuals()
        if len(all_inds) <= self.max_size:
            return
        
        # Sort bins by visit count and remove from least visited
        sorted_bins = sorted(self.cells.keys(), key=lambda b: self.bin_visits[b])
        for bin_coords in sorted_bins:
            if self.total_individuals() <= self.max_size:
                break
            del self.cells[bin_coords]
    
    def get_all_individuals(self) -> List[Individual]:
        """Get all individuals in archive."""
        individuals = []
        for content in self.cells.values():
            individuals.extend(content)
        return individuals
    
    def total_individuals(self) -> int:
        """Total number of individuals."""
        return sum(len(content) for content in self.cells.values())
    
    def size(self) -> int:
        """Number of occupied bins."""
        return len(self.cells)
        """Number of occupied bins."""
        return len(self.cells)
