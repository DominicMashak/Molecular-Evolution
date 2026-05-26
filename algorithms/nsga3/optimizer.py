"""
NSGA-III Optimizer — latent-space genome, pymoo selection.

The genome is a real-valued vector z in the UMAP-reduced transformer
embedding space (ChemBERTa / PolyBERT / MatText).  pymoo's NSGA-III applies
SBX crossover and polynomial mutation directly on z.  Decoding z → SMILES
uses nearest-neighbour lookup in a running pool followed by SMILES mutation
— identical to CMA-MAE.
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


# ---------------------------------------------------------------------------
# JSON encoder (verbatim from cma_mae/optimizer.py)
# ---------------------------------------------------------------------------

class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that converts numpy scalars/arrays to native Python types."""
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
# pymoo Problem wrapper
# ---------------------------------------------------------------------------

class MolecularProblem(Problem):
    """
    Thin wrapper that supplies NSGA-III with variable metadata.

    _evaluate is never called — evaluation is driven entirely by the
    NSGA3Optimizer ask/tell loop.  The bounds xl/xu come from the UMAP
    embedder's fitted range estimates (with margin).
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

class NSGA3Optimizer:
    """
    NSGA-III optimizer over the transformer UMAP embedding space.

    pymoo handles:
      - Das-Dennis reference-point selection
      - Non-dominated sorting
      - SBX crossover + polynomial mutation on z

    This class handles:
      - Decoding z → molecule (NN lookup + SMILES mutation)
      - Fitness evaluation + caching
      - FIFO z_smiles pool management
      - Best-ever molecule database
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
        output_dir: str = "nsga3_results",
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

        # Running pool of (z_vector, smiles) for nearest-neighbour decode.
        # Pre-seeded from the embedder fitting sample — no QC calls needed.
        self._z_smiles_pool: List[Tuple[np.ndarray, str]] = (
            list(initial_pool) if initial_pool else []
        )

        # Evaluation cache (keyed by genotype string)
        self._eval_cache: Dict[str, Dict] = {}

        # Best-ever molecule database (keyed by canonical genotype string)
        self._best_db: Dict[str, Dict] = {}

        # HV indicator (optional — skipped when ref_point is None)
        self._hv_indicator = None
        if hv_ref_point is not None:
            from pymoo.indicators.hv import Hypervolume
            # Convert from original units to pymoo space (negate maximised dims)
            hv_ref_negated = np.array([
                -ref if neg else ref
                for ref, neg in zip(hv_ref_point, negate_mask)
            ])
            self._hv_indicator = Hypervolume(ref_point=hv_ref_negated)

        # Pessimistic fill for failed evaluations (pymoo minimises, so large = bad)
        self._failure_F = np.full(self.n_obj, 1e9)

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.generation = 0
        self.total_evaluations = 0
        self.total_eval_failures = 0
        self.failed_evaluations: List[Dict] = []

        # CSV stats log
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
        """Map a latent vector to a molecule via nearest-neighbour + mutation.

        Verbatim logic from cma_mae/optimizer.py:163-171.
        """
        if not self._z_smiles_pool:
            return self.generate_fn()
        pool_vecs = np.stack([p[0] for p in self._z_smiles_pool])
        dists = np.linalg.norm(pool_vecs - z, axis=1)
        nearest = self._z_smiles_pool[int(np.argmin(dists))][1]
        mutated = self.mutate_fn(nearest)
        return mutated if mutated is not None else nearest

    # ── Evaluation ───────────────────────────────────────────────────────────

    def _evaluate_with_cache(self, smiles: str) -> Dict[str, Any]:
        """Evaluate a molecule, using cache to avoid duplicate QC calls."""
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
        """Append to z_smiles pool with FIFO eviction when full."""
        self._z_smiles_pool.append((z, smiles))
        if len(self._z_smiles_pool) > self.pool_max_size:
            self._z_smiles_pool.pop(0)

    # ── Objective transformation ─────────────────────────────────────────────

    def _objectives_to_pymoo(self, props: Dict) -> np.ndarray:
        """Extract objectives from props dict, applying negate_mask for pymoo.

        pymoo always minimises.  Maximised objectives are negated here so
        the internal pymoo representation is consistent (lower = better).
        The _best_db and saved archives store values in original units.
        """
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
        """Add molecule to _best_db, keeping earliest generation per genotype."""
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
        """Compute hypervolume of F (in pymoo space) vs the configured reference."""
        if self._hv_indicator is None or F_pymoo is None or len(F_pymoo) == 0:
            return 0.0
        try:
            return float(self._hv_indicator(F_pymoo))
        except Exception:
            return 0.0

    # ── Initialise ───────────────────────────────────────────────────────────

    def initialize(self) -> None:
        """Pool is pre-seeded from embedder fitting sample.

        No separate QC evaluation needed before the main loop — the first
        step() call evaluates the NSGA-III initial population.
        """
        print(f"z_smiles pool pre-seeded with {len(self._z_smiles_pool)} "
              "molecules from embedder fitting.")
        print("NSGA-III initial population will be evaluated in step 1.")

    # ── Step ─────────────────────────────────────────────────────────────────

    def step(self) -> Dict[str, Any]:
        """One NSGA-III generation via pymoo ask/tell.

        1. Ask pymoo for offspring z-vectors (real-valued embeddings).
        2. Decode each z → molecule via NN lookup + mutation.
        3. Evaluate molecules and fill objective matrix F.
        4. Tell pymoo (triggers NSGA-III selection + reference-point association).
        5. Collect stats from current Pareto front.
        """
        pop = self.algorithm.ask()
        X = pop.get('X')          # (pop_size, embed_dim)
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

        # Stats from current Pareto front
        opt = self.algorithm.opt
        hv = 0.0
        pareto_size = 0
        if opt is not None and len(opt) > 0:
            opt_F = opt.get('F')
            valid_mask = np.all(opt_F < 1e8, axis=1)
            opt_F_valid = opt_F[valid_mask]
            pareto_size = int(valid_mask.sum())
            hv = self.compute_hv(opt_F_valid)

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
        """Main NSGA-III optimisation loop."""
        self.initialize()
        print(f"\nRunning NSGA-III for {n_generations} generations...")
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
        """Save current NSGA-III Pareto front to JSON in original objective units."""
        opt = self.algorithm.opt
        if opt is None or len(opt) == 0:
            print(f"No Pareto front to save at generation {generation}.")
            return

        X = opt.get('X')
        F = opt.get('F')

        # Reverse lookup: z_bytes → genotype string, built from _best_db
        _z_lookup: Dict[bytes, str] = {}
        for entry in self._best_db.values():
            if entry.get('z_vector'):
                z_arr = np.array(entry['z_vector'])
                _z_lookup[z_arr.tobytes()] = entry.get(self._genotype_key)

        solutions = []
        for i in range(len(X)):
            f_pymoo = F[i]
            if np.any(f_pymoo >= 1e8):
                continue  # skip failure-filled rows

            z = X[i]
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
        """Return the best molecule from _best_db by the first objective."""
        if not self._best_db:
            return None
        key0 = self.objective_keys[0]
        negate0 = self.negate_mask[0]
        valid = [v for v in self._best_db.values() if v.get(key0) is not None]
        if not valid:
            return None
        return (max if negate0 else min)(valid, key=lambda x: x[key0])
