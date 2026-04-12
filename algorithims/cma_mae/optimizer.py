import json
import csv
<<<<<<< Updated upstream
=======
import sys
import os
>>>>>>> Stashed changes
import numpy as np
from pathlib import Path
from typing import Callable, Dict, Any, List, Optional

<<<<<<< Updated upstream
=======
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'molev_utils'))
from diversity_metrics import compute_diversity_metrics


def _morgan_embeddings(smiles_list: List[str], nbits: int = 2048) -> np.ndarray:
    """Compute Morgan fingerprint bit-vectors (ECFP4) for a list of SMILES.
    Returns (N, nbits) float32 array. Invalid SMILES get zero vectors."""
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    out = np.zeros((len(smiles_list), nbits), dtype=np.float32)
    for i, smi in enumerate(smiles_list):
        if smi is None:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nbits)
            out[i] = np.array(fp, dtype=np.float32)
    return out

>>>>>>> Stashed changes

class CMAMaeOptimizer:
    """
    CMA-MAE (Covariance Matrix Adaptation MAP-Annealing) optimizer.

    Uses pyribs for:
    - GridArchive with threshold annealing (learning_rate < 1): the CMA-MAE archive
    - EvolutionStrategyEmitter with improvement ranking ("imp"): the CMA-ES component
    - Scheduler to coordinate emitters and archive

<<<<<<< Updated upstream
    A MolecularVAE provides the continuous latent space:
        CMA-ES generates z ∈ ℝ^latent_dim
        VAE decoder maps z → SELFIES → molecule
        evaluate_fn scores each molecule and returns (objective, measures)
=======
    CMA-ES operates in the frozen ChemBERTa-2 MTR UMAP embedding space.
    The decode step maps a latent vector to a SMILES string via nearest-neighbour
    lookup in the running pool followed by SMILES mutation.
>>>>>>> Stashed changes

    The result_archive (learning_rate=1.0) is a standard MAP-Elites archive that
    tracks the best-ever molecule per cell — used for final output and analysis.
    """

    def __init__(
        self,
        scheduler,
        result_archive,
<<<<<<< Updated upstream
        vae,
=======
        embedder,
        mutate_fn: Callable[[str], Optional[str]],
>>>>>>> Stashed changes
        generate_fn: Callable[[], Optional[str]],
        evaluate_fn: Callable[[str], Dict[str, Any]],
        measure_keys: List[str],
        objective_key: str,
        output_dir: str = "cma_mae_results",
        random_init_size: int = 100,
        threshold_min: float = -float('inf'),
<<<<<<< Updated upstream
    ):
        """
        Args:
            scheduler:        pyribs Scheduler (wraps CMA-MAE archive + emitters)
            result_archive:   pyribs GridArchive with learning_rate=1.0 (best-ever per cell)
            vae:              MolecularVAE — provides encode() and decode()
            generate_fn:      Callable that returns a random SMILES string for initialisation
            evaluate_fn:      Callable that evaluates a SMILES and returns a properties dict
            measure_keys:     Keys in properties dict used as archive measures
            objective_key:    Key in properties dict used as the optimisation objective
            output_dir:       Directory for saving results
            random_init_size: Number of molecules used to initialise the archive
            threshold_min:    Floor for cell thresholds (passed to tell() for failed decodes)
        """
        self.scheduler = scheduler
        self.result_archive = result_archive
        self.vae = vae
=======
        reference_point: Optional[List[float]] = None,
    ):
        self.scheduler = scheduler
        self.result_archive = result_archive
        self.embedder = embedder
        self._embed_dim: int = embedder.n_components
        self.mutate_fn = mutate_fn
>>>>>>> Stashed changes
        self.generate_fn = generate_fn
        self.evaluate_fn = evaluate_fn
        self.measure_keys = measure_keys
        self.objective_key = objective_key
        self.threshold_min = threshold_min
        self.random_init_size = random_init_size
<<<<<<< Updated upstream
=======
        self.reference_point = reference_point if reference_point is not None else [0.0]

        # Running pool of (embedding, SMILES) for nearest-neighbour decode
        self._z_smiles_pool: List[tuple] = []

        self.molecular_embedder = embedder
>>>>>>> Stashed changes

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.generation = 0
        self.total_evaluations = 0
        self.all_molecules: List[Dict] = []
        self._eval_cache: Dict[str, Dict] = {}

<<<<<<< Updated upstream
=======
        # Failed evaluation tracking
        self.total_eval_failures = 0
        self.failed_evaluations: List[Dict] = []

>>>>>>> Stashed changes
        # CSV stats log
        self._stats_file = self.output_dir / 'stats_log.csv'
        with open(self._stats_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'generation', 'coverage', 'archive_size',
<<<<<<< Updated upstream
                'max_objective', 'mean_objective',
                'decode_success_rate', 'total_evaluations',
            ])

=======
                'max_objective', 'mean_objective', 'qd_score',
                'decode_success_rate', 'total_evaluations',
                'int_div', 'scaffold_count', 'n_unique',
                'eval_fail_count', 'eval_fail_rate',
            ])

    def _record_failure(self, smiles: str, props: Dict, generation: int) -> None:
        """Log a failed evaluation (xTB/QC error) without inserting into archive."""
        self.total_eval_failures += 1
        self.failed_evaluations.append({
            'smiles': smiles,
            'generation': generation,
            'error': props.get('error', 'unknown'),
        })

>>>>>>> Stashed changes
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
        if obj is None:
            obj = self.threshold_min
        else:
            obj = float(obj)
        measures = []
        for k in self.measure_keys:
            v = props.get(k)
            measures.append(float(v) if v is not None else 0.0)
        return obj, measures

    def _add_to_archives(self, z: np.ndarray, obj: float, measures: List[float],
                         smiles: str, props: Dict, generation: int):
        """Add a solution to both archives and the molecule database."""
        self.update_molecule_database(smiles, props, z, generation)
        measures_arr = np.array(measures)
        # result_archive: best-ever per cell (standard MAP-Elites)
        try:
            self.result_archive.add_single(z, obj, measures_arr)
        except Exception:
            pass  # measure out of range — skip

<<<<<<< Updated upstream
    def initialize(self) -> None:
        """
        Seed both archives using molecules generated by generate_fn,
        then encoded into latent space via the VAE.
        """
=======
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
        """Seed both archives using random molecules encoded into the ChemBERTa-UMAP space."""
>>>>>>> Stashed changes
        print(f"Initialising archive with {self.random_init_size} molecules...")
        n_added = 0

        for i in range(self.random_init_size):
            smiles = self.generate_fn()
            if smiles is None:
                continue

<<<<<<< Updated upstream
            z = self.vae.encode(smiles)
            if z is None:
                # Fall back to random z if encoding fails
                z = np.random.randn(self.vae.latent_dim).astype(np.float64)
                smiles = self.vae.decode(z)
                if smiles is None:
                    continue

            props = self._evaluate_with_cache(smiles)
=======
            try:
                z = self.embedder.embed([smiles])[0].astype(np.float64)
            except Exception:
                z = np.random.randn(self._embed_dim).astype(np.float64)
            self._z_smiles_pool.append((z, smiles))

            props = self._evaluate_with_cache(smiles)
            if props.get('error') is not None:
                self._record_failure(smiles, props, generation=0)
                continue

>>>>>>> Stashed changes
            obj, measures = self._get_obj_and_measures(props)

            self._add_to_archives(z, obj, measures, smiles, props, generation=0)

            # Also seed the CMA-MAE archive directly so emitters have initial data
            try:
                self.scheduler.archive.add_single(z, obj, np.array(measures))
            except Exception:
                pass

            n_added += 1
            if (i + 1) % max(1, self.random_init_size // 5) == 0:
                coverage = self.result_archive.stats.coverage
                print(f"  {i+1}/{self.random_init_size} — coverage: {coverage:.2%}")

        print(f"Initialisation complete. {n_added} solutions added.")

    def step(self) -> Dict[str, Any]:
        """
        One CMA-MAE generation:
          1. Ask pyribs for a batch of latent vectors (CMA-ES sampling)
          2. Decode each z → SMILES via VAE
          3. Evaluate each molecule
          4. Tell pyribs objectives + measures (updates CMA-MAE thresholds + emitters)
          5. Update result_archive with best-ever logic
        """
        solutions = self.scheduler.ask()  # (total_batch, latent_dim)
        n = len(solutions)

        objectives = np.full(n, self.threshold_min)
        measures = np.zeros((n, len(self.measure_keys)))
        n_decoded = 0

<<<<<<< Updated upstream
        for i, z in enumerate(solutions):
            smiles = self.vae.decode(z)
            if smiles is None:
                continue

            props = self._evaluate_with_cache(smiles)
=======
        # ── Decode all candidates (nearest-neighbour + mutation) ──────────
        decoded = []
        for z in solutions:
            decoded.append(self._decode_to_smiles(z))

        selected_idx = set(range(n))

        for i, (z, smiles) in enumerate(zip(solutions, decoded)):
            if smiles is None:
                continue
            if i not in selected_idx:
                continue

            props = self._evaluate_with_cache(smiles)
            if props.get('error') is not None:
                self._record_failure(smiles, props, generation=self.generation + 1)
                continue

>>>>>>> Stashed changes
            obj, meas = self._get_obj_and_measures(props)

            objectives[i] = obj
            measures[i] = meas
            n_decoded += 1

            self._add_to_archives(z, obj, meas, smiles, props, generation=self.generation + 1)
<<<<<<< Updated upstream
=======
            self._z_smiles_pool.append((z.astype(np.float64), smiles))
>>>>>>> Stashed changes

        # Update CMA-MAE archive and emitter covariance matrices
        self.scheduler.tell(objectives, measures)
        self.generation += 1

        decode_rate = n_decoded / n if n > 0 else 0.0
<<<<<<< Updated upstream
        stats = self.result_archive.stats
=======
        fail_rate = (
            self.total_eval_failures / self.total_evaluations
            if self.total_evaluations > 0 else 0.0
        )
        stats = self.result_archive.stats

        archive_smiles = [m['smiles'] for m in self.all_molecules if m.get('smiles')]
        div = compute_diversity_metrics(archive_smiles, max_sample=500)

        qd_score = float(self.result_archive.stats.qd_score) \
            if self.result_archive.stats.qd_score is not None else 0.0
>>>>>>> Stashed changes
        return {
            'generation': self.generation,
            'coverage': float(stats.coverage),
            'archive_size': int(stats.num_elites),
            'max_objective': float(stats.obj_max) if stats.obj_max is not None else 0.0,
            'mean_objective': float(stats.obj_mean) if stats.obj_mean is not None else 0.0,
<<<<<<< Updated upstream
            'decode_success_rate': decode_rate,
            'total_evaluations': self.total_evaluations,
=======
            'qd_score': qd_score,
            'decode_success_rate': decode_rate,
            'total_evaluations': self.total_evaluations,
            'int_div': div['int_div'],
            'scaffold_count': div['scaffold_count'],
            'n_unique': div['n_unique'],
            'eval_fail_count': self.total_eval_failures,
            'eval_fail_rate': fail_rate,
>>>>>>> Stashed changes
        }

    def _log_stats(self, stats: Dict):
        with open(self._stats_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                stats['generation'], stats['coverage'], stats['archive_size'],
<<<<<<< Updated upstream
                stats['max_objective'], stats['mean_objective'],
                stats['decode_success_rate'], stats['total_evaluations'],
=======
                stats['max_objective'], stats['mean_objective'], stats['qd_score'],
                stats['decode_success_rate'], stats['total_evaluations'],
                stats['int_div'], stats['scaffold_count'], stats['n_unique'],
                stats['eval_fail_count'], stats['eval_fail_rate'],
>>>>>>> Stashed changes
            ])

    def run(
        self,
        n_generations: int,
        log_frequency: int = 10,
        save_frequency: int = 50,
    ) -> List[Dict[str, Any]]:
        """Main CMA-MAE optimisation loop."""
        if self.result_archive.stats.num_elites == 0:
            self.initialize()

        print(f"\nRunning CMA-MAE for {n_generations} generations...")
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
<<<<<<< Updated upstream
                    f"Mean={stats['mean_objective']:8.3f}, "
=======
                    f"QD={stats['qd_score']:10.3f}, "
>>>>>>> Stashed changes
                    f"Decode={stats['decode_success_rate']:5.1%}"
                )

        print("\nOptimisation complete!")
        print(f"Final coverage: {self.result_archive.stats.coverage:.2%}")
        print(f"Final archive size: {self.result_archive.stats.num_elites}")
        print(f"Total evaluations: {self.total_evaluations}")
<<<<<<< Updated upstream
=======
        if self.total_eval_failures > 0:
            fail_rate = self.total_eval_failures / self.total_evaluations
            print(f"Eval failures:     {self.total_eval_failures} ({fail_rate:.1%})")
>>>>>>> Stashed changes

        self.save_archive(n_generations)
        self.save_molecule_database()
        return history

    def update_molecule_database(self, smiles: str, props: Dict,
                                 z: np.ndarray, generation: int):
        """Add or update a molecule in the database (keeps earliest generation)."""
        existing = next((m for m in self.all_molecules if m['smiles'] == smiles), None)
        if existing:
            if generation < existing['generation']:
                existing['generation'] = generation
        else:
            entry: Dict = {'smiles': smiles, 'generation': generation,
                           'z_vector': z.tolist()}
            for k, v in props.items():
                if k not in entry:
                    entry[k] = v
            self.all_molecules.append(entry)

    def save_molecule_database(self):
        db_file = self.output_dir / 'all_molecules_database.json'
        sorted_mols = sorted(self.all_molecules, key=lambda x: x.get('generation', 0))
        with open(db_file, 'w') as f:
            json.dump(sorted_mols, f, indent=2)
        print(f"Saved {len(sorted_mols)} molecules to {db_file}")

<<<<<<< Updated upstream
    def save_archive(self, generation: int):
        """Save current result_archive to JSON. Molecules looked up from database."""
        archive_data = {'generation': generation, 'solutions': []}

        df = self.result_archive.as_pandas(include_solutions=True)
        for _, row in df.iterrows():
            # Decode z to recover SMILES for the archive snapshot
            z = np.array([row[f'solution_{i}'] for i in range(self.vae.latent_dim)])
            smiles = self.vae.decode(z)
=======
        if self.failed_evaluations:
            fail_file = self.output_dir / 'failed_evaluations.json'
            with open(fail_file, 'w') as f:
                json.dump(self.failed_evaluations, f, indent=2)
            print(f"Saved {len(self.failed_evaluations)} failed evaluations to {fail_file}")

    def save_archive(self, generation: int):
        """Save current result_archive to JSON. SMILES looked up from molecule database."""
        archive_data = {'generation': generation, 'solutions': []}

        # Build reverse map from z-bytes → smiles using the molecule database
        _z_lookup = {
            np.array(m['z_vector']).tobytes(): m['smiles']
            for m in self.all_molecules
            if m.get('z_vector') and m.get('smiles')
        }

        df = self.result_archive.data(return_type='pandas')
        for _, row in df.iterrows():
            z = np.array([row[f'solution_{i}'] for i in range(self._embed_dim)])
            smiles = _z_lookup.get(z.tobytes())
>>>>>>> Stashed changes
            entry = {
                'objective': float(row['objective']),
                'measures': [float(row[f'measures_{i}'])
                             for i in range(len(self.measure_keys))],
                'smiles': smiles,
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
        stats = self.result_archive.stats
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
        Creates a standard grid archive snapshot per generation and computes
        coverage and QD score without re-running QC evaluations.
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
        print(f"Rebuilding archive for {max_gen+1} generations...")

        stats_rows = []
        for gen in range(max_gen + 1):
            arc = GridArchive(
                solution_dim=1,  # placeholder
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
