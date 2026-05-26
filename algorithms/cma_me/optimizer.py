import json
import csv
import sys
import os
import numpy as np
from pathlib import Path
from typing import Callable, Dict, Any, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'molev_utils'))
from diversity_metrics import compute_diversity_metrics


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


class CMAMeOptimizer:
    """
    CMA-ME (Covariance Matrix Adaptation MAP-Elites) optimizer.

    Uses pyribs for:
    - GridArchive or CVTArchive with standard MAP-Elites (learning_rate=1.0)
    - EvolutionStrategyEmitter with two-stage improvement ranking ("2imp") and
      filter selection: the classic CMA-ME configuration (Fontaine et al. 2020)
    - Scheduler to coordinate emitters and archive

    CMA-ES operates in the frozen ChemBERTa-2 MTR UMAP embedding space.
    The decode step maps a latent vector to a SMILES string via nearest-neighbour
    lookup in the running pool followed by SMILES mutation.

    Unlike CMA-MAE, CMA-ME uses a single archive with no threshold annealing.
    Cells are updated only when a strictly improving solution is found.
    """

    def __init__(
        self,
        scheduler,
        embedder,
        mutate_fn: Callable[[str], Optional[str]],
        generate_fn: Callable[[], Optional[str]],
        evaluate_fn: Callable[[str], Dict[str, Any]],
        measure_keys: List[str],
        objective_key: str,
        output_dir: str = "cma_me_results",
        random_init_size: int = 100,
        reference_point: Optional[List[float]] = None,
        encoding: str = 'smiles',
    ):
        self.scheduler = scheduler
        self.archive = scheduler.archive  # single archive — no dual archive in CMA-ME
        self.embedder = embedder
        self._embed_dim: int = embedder.n_components
        self.mutate_fn = mutate_fn
        self.generate_fn = generate_fn
        self.evaluate_fn = evaluate_fn
        self.measure_keys = measure_keys
        self.objective_key = objective_key
        self.random_init_size = random_init_size
        self.reference_point = reference_point if reference_point is not None else [0.0]
        self.encoding = encoding
        # Key used to look up the genotype string in props dicts
        self._genotype_key = 'slices' if encoding == 'slices' else 'smiles'

        # Running pool of (embedding, SMILES) for nearest-neighbour decode
        self._z_smiles_pool: List[tuple] = []

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.generation = 0
        self.total_evaluations = 0
        self.all_molecules: List[Dict] = []
        self._eval_cache: Dict[str, Dict] = {}

        # Failed evaluation tracking
        self.total_eval_failures = 0
        self.failed_evaluations: List[Dict] = []

        # CSV stats log
        self._stats_file = self.output_dir / 'stats_log.csv'
        with open(self._stats_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'generation', 'coverage', 'archive_size',
                'max_objective', 'mean_objective', 'qd_score',
                'decode_success_rate', 'total_evaluations',
                'int_div', 'scaffold_count', 'n_unique',
                'eval_fail_count', 'eval_fail_rate',
            ])

    def _record_failure(self, smiles: str, props: Dict, generation: int) -> None:
        """Log a failed evaluation (xTB/QC error) without inserting into archive."""
        self.total_eval_failures += 1
        self.failed_evaluations.append({
            self._genotype_key: smiles,
            'generation': generation,
            'error': props.get('error', 'unknown'),
        })

    def _evaluate_with_cache(self, smiles: str) -> Dict[str, Any]:
        if smiles in self._eval_cache:
            return self._eval_cache[smiles]
        props = self.evaluate_fn(smiles)
        self._eval_cache[smiles] = props
        self.total_evaluations += 1
        return props

    def _get_obj_and_measures(self, props: Dict) -> tuple:
        """Extract (objective float, measures list) from a properties dict."""
        obj = props.get(self.objective_key)
        obj = float(obj) if obj is not None else -float('inf')
        measures = [float(props.get(k) or 0.0) for k in self.measure_keys]
        return obj, measures

    def _add_to_archive(self, z: np.ndarray, obj: float, measures: List[float],
                        smiles: str, props: Dict, generation: int):
        """Add a solution to the archive and molecule database."""
        self.update_molecule_database(smiles, props, z, generation)
        try:
            self.archive.add_single(z, obj, np.array(measures))
        except Exception:
            pass  # measure out of range — skip

    def _decode_to_smiles(self, z: np.ndarray) -> Optional[str]:
        """Map a CMA-ES latent vector to a SMILES string via nearest-neighbour + mutation."""
        if not self._z_smiles_pool:
            return self.generate_fn()
        pool_vecs = np.stack([p[0] for p in self._z_smiles_pool])
        dists = np.linalg.norm(pool_vecs - z, axis=1)
        nearest_smiles = self._z_smiles_pool[int(np.argmin(dists))][1]
        mutated = self.mutate_fn(nearest_smiles)
        return mutated if mutated is not None else nearest_smiles

    def initialize(self) -> None:
        """Seed the archive with randomly generated molecules.

        Keeps generating candidates until at least 1 successful evaluation lands in the
        archive (required for CMA-ME emitters to sample from). Tries up to
        max(random_init_size * 10, 100) total attempts.
        """
        print(f"Initialising archive with {self.random_init_size} molecules...")
        n_added = 0
        max_attempts = max(self.random_init_size * 10, 100)

        i = 0
        attempt = 0
        while i < self.random_init_size or (n_added == 0 and attempt < max_attempts):
            attempt += 1
            smiles = self.generate_fn()
            if smiles is None:
                continue

            try:
                z = self.embedder.embed([smiles])[0].astype(np.float64)
            except Exception:
                z = np.random.randn(self._embed_dim).astype(np.float64)
            self._z_smiles_pool.append((z, smiles))

            props = self._evaluate_with_cache(smiles)
            if props.get('error') is not None:
                self._record_failure(smiles, props, generation=0)
                i += 1
                if i % max(1, self.random_init_size // 5) == 0:
                    coverage = self.archive.stats.coverage
                    print(f"  {i}/{self.random_init_size} — coverage: {coverage:.2%}")
                continue

            obj, measures = self._get_obj_and_measures(props)
            self._add_to_archive(z, obj, measures, smiles, props, generation=0)
            n_added += 1
            i += 1

            if i % max(1, self.random_init_size // 5) == 0:
                coverage = self.archive.stats.coverage
                print(f"  {i}/{self.random_init_size} — coverage: {coverage:.2%}")

        if n_added == 0:
            print(f"WARNING: Initialisation complete but archive is still empty after "
                  f"{attempt} attempts. CMA-ME emitters will use random mean vectors.")
        else:
            print(f"Initialisation complete. {n_added} solutions added ({attempt} attempts).")

    def step(self) -> Dict[str, Any]:
        """
        One CMA-ME generation:
          1. Ask pyribs for a batch of latent vectors (CMA-ES sampling)
          2. Decode each z → SMILES via nearest-neighbour + mutation
          3. Evaluate each molecule
          4. Tell pyribs objectives + measures (updates CMA-ME emitters via 2imp ranking)
        """
        solutions = self.scheduler.ask()  # (total_batch, latent_dim)
        n = len(solutions)

        objectives = np.full(n, -float('inf'))
        measures = np.zeros((n, len(self.measure_keys)))
        n_decoded = 0

        decoded = [self._decode_to_smiles(z) for z in solutions]

        for i, (z, smiles) in enumerate(zip(solutions, decoded)):
            if smiles is None:
                continue

            props = self._evaluate_with_cache(smiles)
            if props.get('error') is not None:
                self._record_failure(smiles, props, generation=self.generation + 1)
                continue

            obj, meas = self._get_obj_and_measures(props)

            objectives[i] = obj
            measures[i] = meas
            n_decoded += 1

            self._add_to_archive(z, obj, meas, smiles, props, generation=self.generation + 1)
            self._z_smiles_pool.append((z.astype(np.float64), smiles))

        # Guard against IndexError when archive is empty (emitter can't sample_elites).
        try:
            self.scheduler.tell(objectives, measures)
        except IndexError:
            pass  # Archive empty — skip emitter mean update this generation
        self.generation += 1

        decode_rate = n_decoded / n if n > 0 else 0.0
        fail_rate = (
            self.total_eval_failures / self.total_evaluations
            if self.total_evaluations > 0 else 0.0
        )
        stats = self.archive.stats

        if self.encoding == 'slices':
            archive_smiles = []  # SLICES are not valid SMILES — skip Morgan diversity
        else:
            archive_smiles = [m['smiles'] for m in self.all_molecules if m.get('smiles')]
        div = compute_diversity_metrics(archive_smiles, max_sample=500)

        qd_score = float(stats.qd_score) if stats.qd_score is not None else 0.0
        return {
            'generation': self.generation,
            'coverage': float(stats.coverage),
            'archive_size': int(stats.num_elites),
            'max_objective': float(stats.obj_max) if stats.obj_max is not None else 0.0,
            'mean_objective': float(stats.obj_mean) if stats.obj_mean is not None else 0.0,
            'qd_score': qd_score,
            'decode_success_rate': decode_rate,
            'total_evaluations': self.total_evaluations,
            'int_div': div['int_div'],
            'scaffold_count': div['scaffold_count'],
            'n_unique': div['n_unique'],
            'eval_fail_count': self.total_eval_failures,
            'eval_fail_rate': fail_rate,
        }

    def _log_stats(self, stats: Dict):
        with open(self._stats_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                stats['generation'], stats['coverage'], stats['archive_size'],
                stats['max_objective'], stats['mean_objective'], stats['qd_score'],
                stats['decode_success_rate'], stats['total_evaluations'],
                stats['int_div'], stats['scaffold_count'], stats['n_unique'],
                stats['eval_fail_count'], stats['eval_fail_rate'],
            ])

    def run(
        self,
        n_generations: int,
        log_frequency: int = 10,
        save_frequency: int = 50,
    ) -> List[Dict[str, Any]]:
        """Main CMA-ME optimisation loop."""
        if self.archive.stats.num_elites == 0:
            self.initialize()

        print(f"\nRunning CMA-ME for {n_generations} generations...")
        history = []

        for gen in range(n_generations):
            stats = self.step()
            self._log_stats(stats)

            if (gen + 1) % save_frequency == 0:
                self.save_archive(gen + 1)
                self.save_molecule_database()

            history.append(stats)

            if (gen + 1) % log_frequency == 0:
                print(
                    f"Gen {stats['generation']:4d}: "
                    f"Coverage={stats['coverage']:6.2%}, "
                    f"Size={stats['archive_size']:5d}, "
                    f"Max={stats['max_objective']:8.3f}, "
                    f"QD={stats['qd_score']:10.3f}, "
                    f"Decode={stats['decode_success_rate']:5.1%}"
                )

        print("\nOptimisation complete!")
        print(f"Final coverage: {self.archive.stats.coverage:.2%}")
        print(f"Final archive size: {self.archive.stats.num_elites}")
        print(f"Total evaluations: {self.total_evaluations}")
        if self.total_eval_failures > 0:
            fail_rate = self.total_eval_failures / self.total_evaluations
            print(f"Eval failures:     {self.total_eval_failures} ({fail_rate:.1%})")

        self.save_archive(n_generations)
        self.save_molecule_database()
        return history

    def update_molecule_database(self, smiles: str, props: Dict,
                                 z: np.ndarray, generation: int):
        """Add a molecule to the database (keeps earliest generation per SMILES)."""
        existing = next((m for m in self.all_molecules if m.get(self._genotype_key) == smiles), None)
        if existing:
            if generation < existing['generation']:
                existing['generation'] = generation
        else:
            entry: Dict = {self._genotype_key: smiles, 'generation': generation,
                           'z_vector': z.tolist()}
            for k, v in props.items():
                if k not in entry:
                    entry[k] = v
            self.all_molecules.append(entry)

    def save_molecule_database(self):
        db_file = self.output_dir / 'all_molecules_database.json'
        sorted_mols = sorted(self.all_molecules, key=lambda x: x.get('generation', 0))
        with open(db_file, 'w') as f:
            json.dump(sorted_mols, f, indent=2, cls=_NumpyEncoder)
        print(f"Saved {len(sorted_mols)} molecules to {db_file}")

        if self.failed_evaluations:
            fail_file = self.output_dir / 'failed_evaluations.json'
            with open(fail_file, 'w') as f:
                json.dump(self.failed_evaluations, f, indent=2, cls=_NumpyEncoder)
            print(f"Saved {len(self.failed_evaluations)} failed evaluations to {fail_file}")

    def save_archive(self, generation: int):
        """Save current archive to JSON with SMILES looked up from molecule database."""
        archive_data = {'generation': generation, 'solutions': []}

        _z_lookup = {
            np.array(m['z_vector']).tobytes(): m.get(self._genotype_key)
            for m in self.all_molecules
            if m.get('z_vector') and m.get(self._genotype_key)
        }

        df = self.archive.data(return_type='pandas')
        for _, row in df.iterrows():
            z = np.array([row[f'solution_{i}'] for i in range(self._embed_dim)])
            smiles = _z_lookup.get(z.tobytes())
            entry = {
                'objective': float(row['objective']),
                'measures': [float(row[f'measures_{i}'])
                             for i in range(len(self.measure_keys))],
                self._genotype_key: smiles,
            }
            archive_data['solutions'].append(entry)

        filename = self.output_dir / f'archive_gen_{generation:04d}.json'
        with open(filename, 'w') as f:
            json.dump(archive_data, f, indent=2)
        print(f"Saved archive ({len(archive_data['solutions'])} cells) to {filename}")

    def get_best_solution(self) -> Optional[Dict]:
        """Return the molecule with the highest objective value from the database."""
        valid = [m for m in self.all_molecules
                 if m.get(self.objective_key) is not None]
        if not valid:
            return None
        return max(valid, key=lambda x: x[self.objective_key])

    def get_statistics(self) -> Dict[str, Any]:
        stats = self.archive.stats
        return {
            'generation': self.generation,
            'coverage': float(stats.coverage),
            'archive_size': int(stats.num_elites),
            'max_objective': float(stats.obj_max) if stats.obj_max is not None else 0.0,
            'mean_objective': float(stats.obj_mean) if stats.obj_mean is not None else 0.0,
            'total_evaluations': self.total_evaluations,
        }

    @staticmethod
    def recalculate_from_database(results_dir: str, archive_config: Dict = None):
        """
        Rebuild archive metrics from an existing all_molecules_database.json.
        Computes coverage and QD score without re-running QC evaluations.
        """
        import json
        from pathlib import Path

        results_path = Path(results_dir)
        db_file = results_path / 'all_molecules_database.json'
        if not db_file.exists():
            raise FileNotFoundError(f"Database not found: {db_file}")

        with open(db_file) as f:
            molecules = json.load(f)
        print(f"Loaded {len(molecules)} molecules from {db_file}")

        if archive_config is None:
            archive_config = {
                'measure_keys': ['num_atoms', 'num_bonds'],
                'measure_ranges': [(1, 50), (0, 60)],
                'dims': [10, 10],
                'objective_key': 'qed',
            }

        dims = archive_config.get('dims', [10, 10])
        measure_keys = archive_config['measure_keys']
        measure_ranges = archive_config['measure_ranges']
        objective_key = archive_config.get('objective_key', 'qed')

        from ribs.archives import GridArchive

        recalc_dir = results_path / 'recalculated'
        recalc_dir.mkdir(exist_ok=True)

        max_gen = max(m.get('generation', 0) for m in molecules)
        print(f"Rebuilding archive for {max_gen + 1} generations...")

        stats_rows = []
        for gen in range(max_gen + 1):
            arc = GridArchive(
                solution_dim=1,
                dims=dims,
                ranges=measure_ranges,
            )
            for mol in molecules:
                if mol.get('generation', 0) > gen:
                    continue
                obj = mol.get(objective_key)
                if obj is None:
                    continue
                meas = [mol.get(k, 0.0) for k in measure_keys]
                try:
                    arc.add_single(np.zeros(1), float(obj), np.array(meas))
                except Exception:
                    pass
            s = arc.stats
            stats_rows.append({
                'generation': gen,
                'coverage': s.coverage,
                'num_elites': s.num_elites,
                'max_objective': s.obj_max,
                'mean_objective': s.obj_mean,
            })
            if gen % 10 == 0 or gen == max_gen:
                print(f"  Gen {gen}: coverage={s.coverage:.2%}, size={s.num_elites}")

        out_file = recalc_dir / 'recalculated_stats.json'
        with open(out_file, 'w') as f:
            json.dump(stats_rows, f, indent=2)
        print(f"Recalculation complete. Stats saved to {out_file}")
