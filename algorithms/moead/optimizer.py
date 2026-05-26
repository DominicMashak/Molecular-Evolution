"""
MOEA/D Optimizer — latent-space genome, pymoo decomposition-based selection.

Uses ParallelMOEAD (batch offspring per generation, identical ask/tell interface
to NSGA-III).  Decomposes objectives into scalar subproblems via Tchebicheff
or PBI scalarization over weight vectors.  The genome is a real-valued vector
z in the UMAP-reduced transformer embedding space; decoding uses
nearest-neighbour lookup + SMILES mutation, identical to CMA-MAE / NSGA-III.

Key difference from NSGA-III: algorithm.opt is the full feasible population,
NOT the Pareto front.  The Pareto front is extracted manually via
find_non_dominated() on algorithm.pop for HV and archive reporting.
"""

import json
import csv
import sys
import os
import numpy as np
from pathlib import Path
from typing import Callable, Dict, Any, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'molev_utils'))
from diversity_metrics import compute_diversity_metrics

from pymoo.core.problem import Problem
from pymoo.util.nds.non_dominated_sorting import find_non_dominated


# ---------------------------------------------------------------------------
# JSON encoder (verbatim from cma_mae/optimizer.py)
# ---------------------------------------------------------------------------

class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return super().default(obj)


# ---------------------------------------------------------------------------
# pymoo Problem wrapper (verbatim from nsga3/optimizer.py)
# ---------------------------------------------------------------------------

class MolecularProblem(Problem):
    """
    Thin wrapper that supplies MOEA/D with variable metadata.
    _evaluate is never called — evaluation is driven by the ask/tell loop.
    """

    def __init__(
        self,
        n_obj: int,
        embed_dim: int,
        z_bounds: List[Tuple[float, float]],
    ):
        xl = np.array([lo for lo, hi in z_bounds])
        xu = np.array([hi for lo, hi in z_bounds])
        super().__init__(n_var=embed_dim, n_obj=n_obj, n_ieq_constr=0,
                         xl=xl, xu=xu)

    def _evaluate(self, X, out, *args, **kwargs):
        raise RuntimeError(
            "MolecularProblem._evaluate should never be called in ask/tell mode."
        )


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

class MOEADOptimizer:
    """
    MOEA/D optimizer over the transformer UMAP embedding space.

    Uses ParallelMOEAD (batch mode) so each generation evaluates pop_size
    offspring, matching the NSGA-III ask/tell pattern.

    pymoo handles:
      - Weight-vector decomposition (Tchebicheff / PBI / WeightedSum)
      - Neighbourhood-based parent selection
      - Ideal-point tracking for scalarization
      - SBX crossover + polynomial mutation on z

    This class handles:
      - Decoding z → molecule (NN lookup + SMILES mutation)
      - Fitness evaluation + caching
      - FIFO z_smiles pool management
      - Best-ever molecule database
      - Pareto front extraction (not provided by algorithm.opt for MOEA/D)
      - Stats logging and checkpointing
    """

    def __init__(
        self,
        algorithm,
        problem: MolecularProblem,
        embedder,
        mutate_fn: Callable[[str], Optional[str]],
        generate_fn: Callable[[], Optional[str]],
        evaluate_fn: Callable[[str], Dict[str, Any]],
        objective_keys: List[str],
        negate_mask: List[bool],
        hv_ref_point: Optional[np.ndarray] = None,
        output_dir: str = "moead_results",
        pool_max_size: int = 10000,
        encoding: str = 'smiles',
        initial_pool: Optional[List[Tuple[np.ndarray, str]]] = None,
    ):
        self.algorithm = algorithm
        self.problem = problem
        self.embedder = embedder
        self._embed_dim: int = problem.n_var
        self.mutate_fn = mutate_fn
        self.generate_fn = generate_fn
        self.evaluate_fn = evaluate_fn
        self.objective_keys = list(objective_keys)
        self.negate_mask = list(negate_mask)
        self.n_obj = len(objective_keys)
        self.pool_max_size = pool_max_size
        self.encoding = encoding
        self._genotype_key = 'slices' if encoding == 'slices' else 'smiles'

        self._z_smiles_pool: List[Tuple[np.ndarray, str]] = (
            list(initial_pool) if initial_pool else []
        )
        self._eval_cache: Dict[str, Dict] = {}
        self._best_db: Dict[str, Dict] = {}

        # HV indicator
        self._hv_indicator = None
        if hv_ref_point is not None:
            from pymoo.indicators.hv import Hypervolume
            hv_ref_negated = np.array([
                -ref if neg else ref
                for ref, neg in zip(hv_ref_point, negate_mask)
            ])
            self._hv_indicator = Hypervolume(ref_point=hv_ref_negated)

        self._failure_F = np.full(self.n_obj, 1e9)

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.generation = 0
        self.total_evaluations = 0
        self.total_eval_failures = 0
        self.failed_evaluations: List[Dict] = []

        self._stats_file = self.output_dir / 'stats_log.csv'
        with open(self._stats_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'generation', 'pareto_size', 'total_db_size',
                'hypervolume', 'decode_success_rate', 'total_evaluations',
                'eval_fail_count', 'eval_fail_rate',
                'int_div', 'scaffold_count', 'n_unique',
            ])

    # ── Decode ───────────────────────────────────────────────────────────────

    def _decode_to_smiles(self, z: np.ndarray) -> Optional[str]:
        """Map latent vector → molecule via nearest-neighbour + mutation."""
        if not self._z_smiles_pool:
            return self.generate_fn()
        pool_vecs = np.stack([p[0] for p in self._z_smiles_pool])
        dists = np.linalg.norm(pool_vecs - z, axis=1)
        nearest = self._z_smiles_pool[int(np.argmin(dists))][1]
        mutated = self.mutate_fn(nearest)
        return mutated if mutated is not None else nearest

    # ── Evaluation ───────────────────────────────────────────────────────────

    def _evaluate_with_cache(self, smiles: str) -> Dict[str, Any]:
        if smiles in self._eval_cache:
            return self._eval_cache[smiles]
        props = self.evaluate_fn(smiles)
        self._eval_cache[smiles] = props
        self.total_evaluations += 1
        return props

    def _record_failure(self, smiles: str, props: Dict, generation: int) -> None:
        self.total_eval_failures += 1
        self.failed_evaluations.append({
            self._genotype_key: smiles,
            'generation': generation,
            'error': props.get('error', 'unknown'),
        })

    # ── Pool management ──────────────────────────────────────────────────────

    def _add_to_pool(self, z: np.ndarray, smiles: str) -> None:
        self._z_smiles_pool.append((z, smiles))
        if len(self._z_smiles_pool) > self.pool_max_size:
            self._z_smiles_pool.pop(0)

    # ── Objective transformation ─────────────────────────────────────────────

    def _objectives_to_pymoo(self, props: Dict) -> np.ndarray:
        """Extract objectives, negate maximised ones for pymoo (minimise all)."""
        F_row = np.empty(self.n_obj)
        for i, (key, negate) in enumerate(zip(self.objective_keys, self.negate_mask)):
            val = props.get(key)
            if val is None:
                F_row[i] = self._failure_F[i]
            else:
                F_row[i] = -float(val) if negate else float(val)
        return F_row

    # ── Database ─────────────────────────────────────────────────────────────

    def _update_best_db(self, smiles: str, props: Dict,
                        z: np.ndarray, gen: int) -> None:
        existing = self._best_db.get(smiles)
        if existing:
            if gen < existing['generation']:
                existing['generation'] = gen
        else:
            entry: Dict = {
                self._genotype_key: smiles,
                'generation': gen,
                'z_vector': z.tolist(),
            }
            for k, v in props.items():
                if k not in entry:
                    entry[k] = v
            self._best_db[smiles] = entry

    # ── Hypervolume ──────────────────────────────────────────────────────────

    def compute_hv(self, F_pymoo: np.ndarray) -> float:
        if self._hv_indicator is None or F_pymoo is None or len(F_pymoo) == 0:
            return 0.0
        try:
            return float(self._hv_indicator(F_pymoo))
        except Exception:
            return 0.0

    # ── Pareto extraction ────────────────────────────────────────────────────

    def _extract_pareto(self) -> Tuple[np.ndarray, int]:
        """Extract Pareto front from current MOEA/D population.

        Unlike NSGA-III (where algorithm.opt IS the Pareto front), MOEA/D's
        algorithm.opt is the full feasible population.  We compute the
        non-dominated set manually.

        Returns (pareto_F, pareto_size) in pymoo space.
        """
        if self.algorithm.pop is None or len(self.algorithm.pop) == 0:
            return np.empty((0, self.n_obj)), 0

        F_pop = self.algorithm.pop.get('F')
        if F_pop is None:
            return np.empty((0, self.n_obj)), 0

        valid_mask = np.all(F_pop < 1e8, axis=1)
        F_valid = F_pop[valid_mask]
        if len(F_valid) == 0:
            return np.empty((0, self.n_obj)), 0

        nd_idx = find_non_dominated(F_valid)
        return F_valid[nd_idx], len(nd_idx)

    # ── Initialise ───────────────────────────────────────────────────────────

    def initialize(self) -> None:
        print(f"z_smiles pool pre-seeded with {len(self._z_smiles_pool)} "
              "molecules from embedder fitting.")
        print("MOEA/D initial population will be evaluated in step 1.")

    # ── Step ─────────────────────────────────────────────────────────────────

    def step(self) -> Dict[str, Any]:
        """One MOEA/D generation via ParallelMOEAD ask/tell.

        ParallelMOEAD returns pop_size offspring per ask() — same cadence as
        NSGA-III.  After tell(), the population is updated via neighbourhood
        replacement based on scalarized (decomposed) fitness.
        """
        pop = self.algorithm.ask()
        X = pop.get('X')
        n = len(X)
        F = np.full((n, self.n_obj), 1e9)
        n_decoded = 0

        for i, z in enumerate(X):
            smiles = self._decode_to_smiles(z)
            if smiles is None:
                continue

            props = self._evaluate_with_cache(smiles)
            if props.get('error') is not None:
                self._record_failure(smiles, props, self.generation + 1)
                continue

            F[i] = self._objectives_to_pymoo(props)
            n_decoded += 1
            self._update_best_db(smiles, props, z.astype(np.float64),
                                 self.generation + 1)
            self._add_to_pool(z.astype(np.float64), smiles)

        pop.set('F', F)
        self.algorithm.tell(infills=pop)
        self.generation += 1

        # Extract Pareto front from population (not algorithm.opt — see docstring)
        pareto_F, pareto_size = self._extract_pareto()
        hv = self.compute_hv(pareto_F)

        decode_rate = n_decoded / max(n, 1)
        fail_rate = self.total_eval_failures / max(self.total_evaluations, 1)

        if self.encoding == 'slices':
            archive_smiles = []
        else:
            archive_smiles = [
                v.get('smiles') for v in self._best_db.values()
                if v.get('smiles')
            ]
        div = compute_diversity_metrics(archive_smiles, max_sample=500)

        return {
            'generation': self.generation,
            'pareto_size': pareto_size,
            'total_db_size': len(self._best_db),
            'hypervolume': hv,
            'decode_success_rate': decode_rate,
            'total_evaluations': self.total_evaluations,
            'eval_fail_count': self.total_eval_failures,
            'eval_fail_rate': fail_rate,
            'int_div': div['int_div'],
            'scaffold_count': div['scaffold_count'],
            'n_unique': div['n_unique'],
        }

    # ── Logging ──────────────────────────────────────────────────────────────

    def _log_stats(self, stats: Dict) -> None:
        with open(self._stats_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                stats['generation'], stats['pareto_size'], stats['total_db_size'],
                stats['hypervolume'], stats['decode_success_rate'],
                stats['total_evaluations'], stats['eval_fail_count'],
                stats['eval_fail_rate'], stats['int_div'],
                stats['scaffold_count'], stats['n_unique'],
            ])

    # ── Main loop ────────────────────────────────────────────────────────────

    def run(
        self,
        n_generations: int,
        log_frequency: int = 10,
        save_frequency: int = 50,
    ) -> List[Dict[str, Any]]:
        """Main MOEA/D optimisation loop."""
        self.initialize()
        print(f"\nRunning MOEA/D for {n_generations} generations...")
        history = []

        for gen in range(n_generations):
            stats = self.step()
            self._log_stats(stats)

            if (gen + 1) % save_frequency == 0:
                self.save_pareto_archive(gen + 1)
                self.save_molecule_database()

            history.append(stats)

            if (gen + 1) % log_frequency == 0:
                print(
                    f"Gen {stats['generation']:4d}: "
                    f"Pareto={stats['pareto_size']:4d}, "
                    f"DB={stats['total_db_size']:5d}, "
                    f"HV={stats['hypervolume']:10.4f}, "
                    f"Decode={stats['decode_success_rate']:5.1%}, "
                    f"Evals={stats['total_evaluations']}"
                )

        print("\nOptimisation complete!")
        print(f"Final Pareto size:  {history[-1]['pareto_size'] if history else 0}")
        print(f"Final DB size:      {len(self._best_db)}")
        print(f"Total evaluations:  {self.total_evaluations}")
        if self.total_eval_failures > 0:
            fail_rate = self.total_eval_failures / max(self.total_evaluations, 1)
            print(f"Eval failures:      {self.total_eval_failures} ({fail_rate:.1%})")

        self.save_pareto_archive(n_generations)
        self.save_molecule_database()
        return history

    # ── Persistence ──────────────────────────────────────────────────────────

    def save_molecule_database(self) -> None:
        db_file = self.output_dir / 'all_molecules_database.json'
        sorted_mols = sorted(
            self._best_db.values(), key=lambda x: x.get('generation', 0)
        )
        with open(db_file, 'w') as f:
            json.dump(sorted_mols, f, indent=2, cls=_NumpyEncoder)
        print(f"Saved {len(sorted_mols)} molecules to {db_file}")

        if self.failed_evaluations:
            fail_file = self.output_dir / 'failed_evaluations.json'
            with open(fail_file, 'w') as f:
                json.dump(self.failed_evaluations, f, indent=2, cls=_NumpyEncoder)
            print(f"Saved {len(self.failed_evaluations)} failures to {fail_file}")

    def save_pareto_archive(self, generation: int) -> None:
        """Save current Pareto front (extracted from MOEA/D population) to JSON."""
        if self.algorithm.pop is None or len(self.algorithm.pop) == 0:
            print(f"No population to save at generation {generation}.")
            return

        X_pop = self.algorithm.pop.get('X')
        F_pop = self.algorithm.pop.get('F')

        valid_mask = np.all(F_pop < 1e8, axis=1)
        F_valid = F_pop[valid_mask]
        X_valid = X_pop[valid_mask]

        if len(F_valid) == 0:
            print(f"No valid solutions to save at generation {generation}.")
            return

        nd_idx = find_non_dominated(F_valid)
        X_nd = X_valid[nd_idx]
        F_nd = F_valid[nd_idx]

        # Reverse lookup z_bytes → genotype string
        _z_lookup: Dict[bytes, str] = {
            np.array(entry['z_vector']).tobytes(): entry.get(self._genotype_key)
            for entry in self._best_db.values()
            if entry.get('z_vector')
        }

        solutions = []
        for i in range(len(X_nd)):
            f_pymoo = F_nd[i]
            z = X_nd[i]
            objectives_original = {
                key: float(-f_pymoo[j] if neg else f_pymoo[j])
                for j, (key, neg) in enumerate(
                    zip(self.objective_keys, self.negate_mask)
                )
            }
            smiles = _z_lookup.get(z.tobytes())
            solutions.append({
                self._genotype_key: smiles,
                'objectives': objectives_original,
                'z_vector': z.tolist(),
            })

        archive_data = {'generation': generation, 'solutions': solutions}
        filename = self.output_dir / f'pareto_archive_gen_{generation:04d}.json'
        with open(filename, 'w') as f:
            json.dump(archive_data, f, indent=2, cls=_NumpyEncoder)
        print(f"Saved Pareto archive ({len(solutions)} solutions) to {filename}")

    def get_best_solution(self) -> Optional[Dict]:
        """Return best molecule from _best_db by the first objective."""
        if not self._best_db:
            return None
        key0 = self.objective_keys[0]
        negate0 = self.negate_mask[0]
        valid = [v for v in self._best_db.values() if v.get(key0) is not None]
        if not valid:
            return None
        return (max if negate0 else min)(valid, key=lambda x: x[key0])
