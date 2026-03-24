import numpy as np
from typing import List, Tuple, Optional, Any, Dict
from scipy.spatial import KDTree
from scipy.cluster.vq import kmeans2

"""
CVT-MAP-Elites Archive: Centroidal Voronoi Tessellation for single-objective MAP-Elites.
Replaces the fixed grid tessellation with CVT for more flexible niche boundaries.
Each Voronoi cell stores a single best solution (same as MAPElitesArchive cell semantics).
"""


class CVTMAPElitesArchive:
    """
    Archive for Quality Diversity optimization using CVT tessellation.
    Stores a single best solution per Voronoi cell defined by k-means centroids.
    Drop-in replacement for MAPElitesArchive with continuous measure support.
    """

    def __init__(self, n_centroids: int, measure_keys: List[str],
                 objective_key: str,
                 measure_bounds: List[Tuple[float, float]],
                 cvt_samples: int = 50000,
                 random_state: int = 42,
                 seed_data: Optional[np.ndarray] = None):
        """
        Initialize the CVT-MAP-Elites archive.

        Args:
            n_centroids: Number of CVT cells (Voronoi regions).
            measure_keys: Keys in properties dict for continuous descriptors
                          (e.g., ['num_atoms', 'num_bonds']).
            objective_key: Key for the objective value to maximize.
            measure_bounds: List of (min, max) for each measure dimension.
            cvt_samples: Number of random samples for k-means centroid generation
                         (only used when seed_data is None).
            random_state: Seed for reproducible centroid placement.
            seed_data: Optional array of shape (N, n_dims) containing real
                       embedding vectors for data-driven centroid placement.
        """
        self.n_centroids = n_centroids
        self.measure_keys = measure_keys
        self.n_dims = len(measure_keys)
        self.objective_key = objective_key
        self.measure_bounds = np.array(measure_bounds, dtype=float)

        if len(measure_keys) != len(measure_bounds):
            raise ValueError(f"Number of measure_keys ({len(measure_keys)}) must match "
                           f"number of measure_bounds ({len(measure_bounds)})")

        # Generate CVT centroids and build KDTree
        self.centroids = self._generate_centroids(cvt_samples, random_state, seed_data)
        self.kd_tree = KDTree(self.centroids)

        # Storage: centroid_index -> {solution, properties, objective} or None
        self.cells: Dict[int, Optional[Dict]] = {
            i: None for i in range(self.n_centroids)
        }

        # Compatibility: measure_dims as tuple so np.prod(archive.measure_dims) works
        self.measure_dims = (self.n_centroids,)

        # Track statistics
        self.n_filled = 0
        self.n_updates = 0

    def _generate_centroids(self, n_samples: int, seed: int,
                            seed_data: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate CVT centroids via k-means.

        When seed_data is provided, runs k-means on real molecule embeddings
        so centroids are placed where molecules actually exist.
        Otherwise falls back to uniform random samples.
        """
        if seed_data is not None:
            print(f"CVT: generating {self.n_centroids} centroids from "
                  f"{len(seed_data)} real molecule embeddings (data-driven).")
            samples = seed_data
        else:
            print(f"CVT: generating {self.n_centroids} centroids from "
                  f"{n_samples} uniform random samples (fallback).")
            rng = np.random.RandomState(seed)
            samples = np.column_stack([
                rng.uniform(lo, hi, size=n_samples)
                for lo, hi in self.measure_bounds
            ])
        centroids, _ = kmeans2(
            samples, self.n_centroids, minit='points', seed=seed
        )
        return centroids

    def _get_niche_index(self, measures: np.ndarray) -> int:
        """Return index of nearest centroid for a continuous measure vector."""
        _, idx = self.kd_tree.query(measures)
        return int(idx)

    def _extract_measures(self, properties: Dict[str, Any]) -> Optional[int]:
        """Extract continuous measure values and return niche (centroid) index."""
        try:
            measures = np.array([float(properties[key]) for key in self.measure_keys])
        except (KeyError, ValueError, TypeError):
            return None

        measures = np.clip(measures, self.measure_bounds[:, 0], self.measure_bounds[:, 1])
        return self._get_niche_index(measures)

    def _extract_objective(self, properties: Dict[str, Any]) -> float:
        """Extract objective value from properties dictionary."""
        try:
            return float(properties[self.objective_key])
        except KeyError:
            raise ValueError(f"Missing required objective key: '{self.objective_key}'")
        except (ValueError, TypeError):
            raise ValueError(f"Invalid objective value for '{self.objective_key}': "
                           f"{properties.get(self.objective_key)}")

    def add(self, solution: Any, properties: Dict[str, Any]) -> bool:
        """
        Add a solution to the archive if it improves the corresponding cell.

        Args:
            solution: The solution object to store (typically a SMILES string)
            properties: Dictionary containing all properties including objective and measures

        Returns:
            True if the solution was added (new cell or improvement), False otherwise
        """
        objective = self._extract_objective(properties)
        niche_idx = self._extract_measures(properties)

        if niche_idx is None:
            return False

        current = self.cells[niche_idx]

        if current is None or objective > current['objective']:
            was_empty = current is None

            self.cells[niche_idx] = {
                'solution': solution,
                'properties': properties.copy(),
                'objective': objective
            }

            if was_empty:
                self.n_filled += 1
            self.n_updates += 1
            return True

        return False

    def get(self, measures: List[float]) -> Optional[Dict[str, Any]]:
        """Retrieve the solution at the cell nearest to the given measures."""
        measures_arr = np.array(measures, dtype=float)
        measures_arr = np.clip(measures_arr, self.measure_bounds[:, 0], self.measure_bounds[:, 1])
        idx = self._get_niche_index(measures_arr)
        return self.cells[idx]

    def get_all_solutions(self) -> List[Dict[str, Any]]:
        """Get all solutions in the archive."""
        results = []
        for idx, cell in self.cells.items():
            if cell is not None:
                results.append({
                    'indices': (idx,),
                    'solution': cell['solution'],
                    'properties': cell['properties'],
                    'objective': cell['objective']
                })
        return results

    def get_coverage(self) -> float:
        """Get the proportion of filled cells."""
        return self.n_filled / self.n_centroids

    def get_max_objective(self) -> float:
        """Get the maximum objective value in the archive."""
        if self.n_filled == 0:
            return -np.inf
        return max(cell['objective'] for cell in self.cells.values() if cell is not None)

    def get_mean_objective(self) -> float:
        """Get the mean objective value of filled cells."""
        if self.n_filled == 0:
            return -np.inf
        filled = [cell['objective'] for cell in self.cells.values() if cell is not None]
        return np.mean(filled)

    def iter_filled_cells(self):
        """Yield (index, cell) for all non-empty cells."""
        for idx, cell in self.cells.items():
            if cell is not None:
                yield (idx,), cell

    def __len__(self) -> int:
        """Return the number of filled cells."""
        return self.n_filled

    def __repr__(self) -> str:
        return (f"CVTMAPElitesArchive(n_centroids={self.n_centroids}, "
                f"measures={self.measure_keys}, objective='{self.objective_key}', "
                f"filled={self.n_filled}/{self.n_centroids}, "
                f"coverage={self.get_coverage():.2%})")
