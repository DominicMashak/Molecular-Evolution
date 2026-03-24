import numpy as np
from typing import List, Tuple, Optional, Any, Dict
from scipy.spatial import KDTree
from scipy.cluster.vq import kmeans2
from pymoo.indicators.hv import Hypervolume

"""
CVT-MOME Archive: Centroidal Voronoi Tessellation for Multi-Objective MAP-Elites
Replaces the fixed grid tessellation with CVT for more flexible niche boundaries.
Each Voronoi cell stores a Pareto front, identical to MOMEArchive cell semantics.
"""


class CVTMOMEArchive:
    """
    Archive for Multi-Objective Quality Diversity optimization using CVT tessellation.
    Stores Pareto fronts in Voronoi cells defined by k-means centroids.
    """

    def __init__(self, n_centroids: int, measure_keys: List[str],
                 objective_keys: List[str],
                 measure_bounds: List[Tuple[float, float]],
                 max_front_size: int = 50,
                 use_crowding_distance: bool = True,
                 optimize_objectives: List[Tuple[str, Any]] = None,
                 cvt_samples: int = 50000,
                 random_state: int = 42,
                 seed_data: Optional[np.ndarray] = None):
        """
        Initialize the CVT-MOME archive.

        Args:
            n_centroids: Number of CVT cells (Voronoi regions).
            measure_keys: Keys in properties dict for continuous descriptors
                          (e.g., ['num_atoms', 'num_bonds']).
            objective_keys: Keys for objective values.
            measure_bounds: List of (min, max) for each measure dimension.
            max_front_size: Maximum number of solutions in each Pareto front.
            use_crowding_distance: If True, use crowding distance for removal.
            optimize_objectives: List of tuples (opt_type, target) where
                                opt_type is 'max', 'min', or 'target'.
            cvt_samples: Number of random samples for k-means centroid generation
                         (only used when seed_data is None).
            random_state: Seed for reproducible centroid placement.
            seed_data: Optional array of shape (N, n_dims) containing real
                       embedding vectors to use as k-means input instead of
                       uniform random samples. When provided, centroids are
                       placed where molecules actually live in the manifold,
                       eliminating empty cells and catch-all cells caused by
                       the mismatch between uniform sampling and the true
                       (non-uniform) molecular embedding distribution.
        """
        self.n_centroids = n_centroids
        self.measure_keys = measure_keys
        self.n_dims = len(measure_keys)
        self.objective_keys = objective_keys
        self.n_objectives = len(objective_keys)
        self.measure_bounds = np.array(measure_bounds, dtype=float)
        self.max_front_size = max_front_size
        self.use_crowding_distance = use_crowding_distance

        if optimize_objectives is None:
            self.optimize_objectives = [('max', None)] * self.n_objectives
        else:
            self.optimize_objectives = optimize_objectives

        if len(measure_keys) != self.n_dims:
            raise ValueError(f"Number of measure_keys ({len(measure_keys)}) must match "
                           f"number of measure_bounds ({len(measure_bounds)})")

        # Generate CVT centroids and build KDTree
        self.centroids = self._generate_centroids(cvt_samples, random_state, seed_data)
        self.kd_tree = KDTree(self.centroids)

        # Fronts stored as dict: centroid_index -> list of solution dicts
        self.fronts: Dict[int, List[Dict]] = {
            i: [] for i in range(self.n_centroids)
        }

        # Compatibility: measure_dims as tuple so np.prod(archive.measure_dims) works
        self.measure_dims = (self.n_centroids,)

        # Track statistics
        self.n_filled = 0
        self.n_updates = 0

        # Initialize PyMoo hypervolume indicator
        self.hv_indicator = Hypervolume(ref_point=self._get_ref_point())

    def _generate_centroids(self, n_samples: int, seed: int,
                            seed_data: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate CVT centroids via k-means.

        When seed_data is provided (real molecule embeddings from the UMAP
        fitting sample), k-means runs on those points so centroids are placed
        in regions of the manifold that molecules actually occupy.

        When seed_data is None, falls back to k-means on uniform random
        samples within measure_bounds (the original behaviour).

        Returns:
            Array of shape (n_centroids, n_dims) with centroid positions.
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
        """
        Extract continuous measure values and return niche (centroid) index.
        Clips to bounds so edge solutions map to nearest edge centroid.
        """
        try:
            measures = np.array([float(properties[key]) for key in self.measure_keys])
        except (KeyError, ValueError, TypeError):
            return None

        measures = np.clip(measures, self.measure_bounds[:, 0], self.measure_bounds[:, 1])
        return self._get_niche_index(measures)

    # ── Objective handling (same as MOMEArchive) ─────────────────

    def _get_ref_point(self):
        """Get reference point for hypervolume calculation in transformed (minimization) space."""
        ref = []
        for opt, _ in self.optimize_objectives:
            if opt == 'max':
                ref.append(0.0)
            else:
                ref.append(1000.0)
        return np.array(ref)

    def _transform_objectives(self, objectives: np.ndarray) -> np.ndarray:
        """Transform objectives for minimization (negate maximization objectives)."""
        transformed = objectives.copy()
        for i, (opt, _) in enumerate(self.optimize_objectives):
            if opt == 'max':
                transformed[:, i] = -transformed[:, i]
        return transformed

    def _extract_objectives(self, properties: Dict[str, Any]) -> np.ndarray:
        """Extract objective values as array."""
        try:
            return np.array([float(properties[key]) for key in self.objective_keys])
        except KeyError as e:
            raise ValueError(f"Missing required objective key: {e}")
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid objective value: {e}")

    # ── Pareto front logic (same as MOMEArchive) ─────────────────

    def _dominates(self, obj1: np.ndarray, obj2: np.ndarray) -> bool:
        """Check if obj1 strictly dominates obj2 (all >= and at least one >)."""
        return np.all(obj1 >= obj2) and np.any(obj1 > obj2)

    def _calculate_crowding_distance(self, front: List[Dict]) -> np.ndarray:
        """
        Calculate crowding distance for each solution in the front.
        Solutions with larger crowding distance are more isolated (less crowded).
        """
        n = len(front)
        if n <= 2:
            return np.full(n, np.inf)

        objectives = np.array([entry['objectives'] for entry in front])
        distances = np.zeros(n)

        for m in range(self.n_objectives):
            sorted_indices = np.argsort(objectives[:, m])
            distances[sorted_indices[0]] = np.inf
            distances[sorted_indices[-1]] = np.inf

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
        """Update a Pareto front with a new solution."""
        # Check if new solution is dominated
        for entry in front:
            if self._dominates(entry['objectives'], new_objectives):
                return front, False

        # Remove dominated solutions
        new_front = []
        for entry in front:
            if not self._dominates(new_objectives, entry['objectives']):
                new_front.append(entry)

        new_front.append({
            'solution': new_solution,
            'properties': new_properties.copy(),
            'objectives': new_objectives
        })

        # Prune if exceeding max size
        if len(new_front) > self.max_front_size:
            if self.use_crowding_distance:
                distances = self._calculate_crowding_distance(new_front)
                finite_indices = np.where(np.isfinite(distances))[0]
                if len(finite_indices) > 0:
                    remove_idx = finite_indices[np.argmin(distances[finite_indices])]
                else:
                    remove_idx = np.random.randint(len(new_front))
            else:
                remove_idx = np.random.randint(len(new_front))
            new_front.pop(remove_idx)

        return new_front, True

    # ── Public interface (same signatures as MOMEArchive) ────────

    def add(self, solution: Any, properties: Dict[str, Any]) -> bool:
        """Add a solution to the archive if it's not dominated in its cell."""
        objectives = self._extract_objectives(properties)
        niche_idx = self._extract_measures(properties)

        if niche_idx is None:
            return False

        current_front = self.fronts[niche_idx]
        was_empty = len(current_front) == 0

        new_front, was_added = self._update_pareto_front(
            current_front, solution, properties, objectives
        )

        if was_added:
            self.fronts[niche_idx] = new_front
            if was_empty and len(new_front) > 0:
                self.n_filled += 1
            elif not was_empty and len(new_front) == 0:
                self.n_filled -= 1
            self.n_updates += 1
            return True
        return False

    def sample_solution_from_archive(self) -> Optional[Any]:
        """Sample uniformly: random filled cell, then random solution from its front."""
        filled = [i for i, f in self.fronts.items() if len(f) > 0]
        if not filled:
            return None
        cell_idx = filled[np.random.randint(len(filled))]
        front = self.fronts[cell_idx]
        return front[np.random.randint(len(front))]['solution']

    def get_front(self, measures: List[float]) -> Optional[List[Dict[str, Any]]]:
        """Retrieve the Pareto front for the cell nearest to the given measures."""
        measures_arr = np.array(measures, dtype=float)
        measures_arr = np.clip(measures_arr, self.measure_bounds[:, 0], self.measure_bounds[:, 1])
        idx = self._get_niche_index(measures_arr)
        front = self.fronts[idx]
        return front if len(front) > 0 else None

    def get_all_solutions(self) -> List[Dict[str, Any]]:
        """Get all solutions in the archive."""
        results = []
        for idx, front in self.fronts.items():
            for entry in front:
                results.append({
                    'indices': (idx,),
                    'solution': entry['solution'],
                    'properties': entry['properties'],
                    'objectives': entry['objectives']
                })
        return results

    def get_coverage(self) -> float:
        """Get the proportion of filled cells."""
        return self.n_filled / self.n_centroids

    def iter_filled_cells(self):
        """Yield (index, front) for all non-empty cells."""
        for idx, front in self.fronts.items():
            if len(front) > 0:
                yield (idx,), front

    # ── Hypervolume and MOQD (same as MOMEArchive) ───────────────

    def _calculate_cell_hypervolume(self, front: List[Dict]) -> float:
        """Calculate hypervolume for a single cell's Pareto front using PyMoo."""
        if not front:
            return 0.0
        objectives = np.array([entry['objectives'] for entry in front])
        transformed_objectives = self._transform_objectives(objectives)
        try:
            return float(self.hv_indicator(transformed_objectives))
        except Exception:
            return 0.0

    def get_hypervolume_per_cell(self) -> Dict[int, float]:
        """Calculate hypervolume for each filled cell's Pareto front."""
        return {
            idx: self._calculate_cell_hypervolume(front)
            for idx, front in self.fronts.items()
            if len(front) > 0
        }

    def compute_moqd_score(self) -> float:
        """Compute MOQD score as sum of hypervolumes of individual cell Pareto fronts."""
        total_score = 0.0
        for front in self.fronts.values():
            if front:
                total_score += self._calculate_cell_hypervolume(front)
        return total_score

    def get_moqd_score(self) -> float:
        """Alias for compute_moqd_score() for backward compatibility."""
        return self.compute_moqd_score()

    def compute_global_hypervolume(self) -> float:
        """Compute global hypervolume by combining all Pareto fronts."""
        global_front = self.get_global_pareto_front()
        if not global_front:
            return 0.0
        objectives = np.array([entry['objectives'] for entry in global_front])
        transformed_objectives = self._transform_objectives(objectives)
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
        return sum(len(f) for f in self.fronts.values())

    def __repr__(self) -> str:
        removal_strategy = "crowding" if self.use_crowding_distance else "random"
        return (f"CVTMOMEArchive(n_centroids={self.n_centroids}, "
                f"measures={self.measure_keys}, objectives={self.objective_keys}, "
                f"filled={self.n_filled}/{self.n_centroids}, "
                f"total_solutions={len(self)}, "
                f"coverage={self.get_coverage():.2%}, "
                f"removal={removal_strategy})")
