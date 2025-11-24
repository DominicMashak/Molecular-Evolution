from typing import Dict, List, Tuple
from collections import defaultdict
import numpy as np
from individual import Individual
from dominance import dominates, fast_non_dominated_sort
from pymoo.indicators.hv import Hypervolume
import matplotlib.pyplot as plt

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
        
        current_front = self.cells.get(bin_coords, [])
        combined = current_front + [individual]
        
        fronts = fast_non_dominated_sort(combined, self.optimize_objectives)
        new_front = fronts[0] if fronts else []
        
        if new_front != current_front:
            self.cells[bin_coords] = new_front
            # Prune if total archive size exceeds max_size
            if self.total_individuals() > self.max_size:
                self._prune_to_size()
            return True
        return False
    
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
    
    def compute_global_hypervolume(self) -> float:
        """Compute global hypervolume by combining all Pareto fronts and calculating HV on the overall front."""
        all_individuals = self.get_all_individuals()
        if not all_individuals:
            return 0.0
        fronts = fast_non_dominated_sort(all_individuals, self.optimize_objectives)
        pareto_front = fronts[0] if fronts else []
        if not pareto_front:
            return 0.0
        objectives = np.array([ind.objectives for ind in pareto_front])
        transformed_objectives = self._transform_objectives(objectives)
        return self.hv_indicator(transformed_objectives)
    
    def compute_moqd_score(self) -> float:
        """Compute MOQD score as sum of hypervolumes of individual cell Pareto fronts."""
        total_score = 0.0
        for bin_coords, front in self.cells.items():
            if front:
                objectives = np.array([ind.objectives for ind in front])
                transformed_objectives = self._transform_objectives(objectives)
                total_score += self.hv_indicator(transformed_objectives)
        return total_score
    
    def plot_individual_cells(self, output_dir):
        """Save individual plots for each cell's Pareto front."""
        for bin_coords, front in self.cells.items():
            if front:
                fig, ax = plt.subplots()
                objectives = np.array([ind.objectives for ind in front])
                if objectives.shape[1] >= 2:
                    ax.scatter(objectives[:, 0], objectives[:, 1], c='red', label='Pareto Front')
                    ax.set_xlabel('Objective 1')
                    ax.set_ylabel('Objective 2')
                    ax.set_title(f'Cell {bin_coords} Pareto Front')
                    ax.legend()
                    plt.savefig(output_dir / f'cell_{bin_coords}.png')
                    plt.close()
