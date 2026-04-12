import json
import csv
import sys
import os
import numpy as np
from pathlib import Path
from typing import Callable, Dict, Any, List, Optional, Tuple

from pymoo.indicators.hv import Hypervolume

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


class MOCMAMaeOptimizer:
    """
    Multi-Objective CMA-MAE (MO-CMA-MAE).

    Implements the two-archive + threshold-front design from:
        Zhao & Nikolaidis, "MO-CMA-MAE", GECCO 2025 (arXiv:2505.20712).

    Two archives:
      - Adaptation archive: pyribs CVTArchive (threshold annealing, learning_rate α).
        Stores latent vectors z and HVI(f, T_e) as the scalar CMA-ES "objective".
      - Result archive: CVTMOMEArchive — true Pareto front F_e per CVT cell.

    Per-cell threshold front T_e (Algorithm 1 from the paper):
      - Separate from F_e; initialised empty per cell.
      - HVI for the CMA-ES signal is HV(T_e ∪ {f}) - HV(T_e).
      - When HVI > 0, bisection finds d_i ∈ (0,1] s.t.
            HVI(d_i*(f_trans - ref) + ref, T_e) = α · HVI(f_trans, T_e)
        and the discounted vector is inserted into T_e (Pareto dominance update).
      - All T_e operations are in the transformed (all-minimisation) space.

    Note: cycle restart rule from the paper is not implemented.
    """

    def __init__(
        self,
        scheduler,
        result_archive,
        embedder,
        mutate_fn: Callable[[str], Optional[str]],
        generate_fn: Callable[[], Optional[str]],
        evaluate_fn: Callable[[str], Dict[str, Any]],
        measure_keys: List[str],
        objective_keys: List[str],
        optimize_objectives: List[Tuple[str, Any]],
        reference_point: np.ndarray,
        output_dir: str = "mo_cma_mae_results",
        random_init_size: int = 100,
        threshold_min: float = 0.0,
        learning_rate: float = 0.01,
        n_centroids: int = 100,
        diversity_log_interval: int = 10,
        step_timing: bool = False,
    ):
        self.scheduler = scheduler
        self.result_archive = result_archive
        self.embedder = embedder
        self._embed_dim: int = embedder.n_components
        self.mutate_fn = mutate_fn
        self.generate_fn = generate_fn
        self.evaluate_fn = evaluate_fn
        self.measure_keys = measure_keys
        self.objective_keys = objective_keys
        self.optimize_objectives = optimize_objectives
        self.reference_point = np.asarray(reference_point, dtype=float)
        self.threshold_min = threshold_min
        self.learning_rate = learning_rate
        self.random_init_size = random_init_size

        # Running pool of (ChemBERTa-UMAP embedding, SMILES) used by _decode_to_smiles.
        # CMA-ES proposals are 10D UMAP coordinates; the pool maps each coordinate to
        # the nearest evaluated molecule which is then mutated.
        self._z_smiles_pool: List[Tuple[np.ndarray, str]] = []

        self.molecular_embedder = embedder
        self.n_centroids = n_centroids
        self.diversity_log_interval = max(1, diversity_log_interval)
        self.step_timing = step_timing
        self._cached_diversity: Dict[str, Any] = {
            'int_div': 0.0, 'scaffold_count': 0, 'n_unique': 0
        }

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.generation = 0
        self.total_evaluations = 0
        self.all_molecules: List[Dict] = []
        self._eval_cache: Dict[str, Dict] = {}

        # PyMoo HV indicator in the all-minimisation transformed space.
        # ref[i] = 0.0 for maximised objectives (their negated values are ≤ 0),
        # ref[i] = 1000.0 for minimised objectives (values are ≤ reference).
        self._ref_point_transformed = self._make_ref_point()
        self._hv = Hypervolume(ref_point=self._ref_point_transformed)

        # Per-cell threshold fronts T_e.
        # Maps cell_index (int) → list of 1-D numpy arrays in transformed space.
        # Empty list means the cell's threshold front has not been initialised yet.
        self.threshold_fronts: Dict[int, List[np.ndarray]] = {}

        # Cycle restart tracking (Section 4.3).
        # Total times each CVT cell has received a decoded solution.
        self._cell_visit_counts: Dict[int, int] = {}
        # Per-emitter: last modal cell visited and how many consecutive generations
        # that cell has been the dominant one.
        n_emitters = len(scheduler.emitters)
        self._emitter_last_cell: List[int] = [-1] * n_emitters
        self._emitter_cell_streak: List[int] = [0] * n_emitters

        # Failed evaluation tracking
        self.total_eval_failures = 0
        self.failed_evaluations: List[Dict] = []

        # CSV stats log
        self._stats_file = self.output_dir / 'stats_log.csv'
        with open(self._stats_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'generation', 'coverage', 'archive_size',
                'moqd_score', 'global_hypervolume',
                'max_hv_contrib', 'mean_hv_contrib',
                'decode_success_rate', 'total_evaluations',
                'int_div', 'scaffold_count', 'n_unique',
                'eval_fail_count', 'eval_fail_rate',
            ])

    # ------------------------------------------------------------------
    # Objective transform helpers
    # ------------------------------------------------------------------

    def _make_ref_point(self) -> np.ndarray:
        """Reference point in transformed (all-minimisation) space.

        Converts the user-supplied reference_point (in original objective space)
        to the all-minimise space used internally for T_e and HV computations.
        For 'max' objectives the value is negated so it remains the worst-case bound.
        """
        ref = self.reference_point.copy().astype(float)
        for i, (opt, _) in enumerate(self.optimize_objectives):
            if opt == 'max':
                ref[i] = -ref[i]
        return ref

    def _transform_objectives(self, objectives: np.ndarray) -> np.ndarray:
        """Convert objectives to all-minimise space (negate maximised objectives).

        Accepts either a 1-D vector (single solution) or a 2-D array (batch).
        """
        t = objectives.copy().astype(float)
        if t.ndim == 1:
            for i, (opt, _) in enumerate(self.optimize_objectives):
                if opt == 'max':
                    t[i] = -t[i]
        else:
            for i, (opt, _) in enumerate(self.optimize_objectives):
                if opt == 'max':
                    t[:, i] = -t[:, i]
        return t

    # ------------------------------------------------------------------
    # Threshold-front helpers (T_e operates in transformed space)
    # ------------------------------------------------------------------

    def _hv_of_trans_front(self, front_trans: List[np.ndarray]) -> float:
        """Hypervolume of a list of already-transformed (all-minimise) objective vectors."""
        if not front_trans:
            return 0.0
        objs = np.array(front_trans)
        try:
            return float(self._hv(objs))
        except Exception:
            return 0.0

    def _dominates_min(self, a: np.ndarray, b: np.ndarray) -> bool:
        """True if a strictly dominates b in minimisation space (all a[i] ≤ b[i], some <)."""
        return bool(np.all(a <= b) and np.any(a < b))

    def _get_cell_idx(self, meas: np.ndarray) -> int:
        """CVT cell index for a measure vector (clips to archive bounds first)."""
        clipped = np.clip(
            meas,
            self.result_archive.measure_bounds[:, 0],
            self.result_archive.measure_bounds[:, 1],
        )
        return self.result_archive._get_niche_index(clipped)

    def _bisect_discount(
        self,
        obj_trans: np.ndarray,
        t_e: List[np.ndarray],
    ) -> float:
        """Bisection search for the discount factor d_i.

        Finds d_i ∈ (0, 1] such that:
            HVI(d_i*(f - ref) + ref, T_e) = α · HVI(f, T_e)

        where α = self.learning_rate.  As d increases from 0→1 the
        interpolated point moves from the reference (HVI=0) to the
        full objective vector (HVI=HVI_full), so the search is monotone.

        Returns 0.0 if the solution does not improve T_e.
        """
        hvi_full = (
            self._hv_of_trans_front(t_e + [obj_trans])
            - self._hv_of_trans_front(t_e)
        )
        if hvi_full <= 1e-12:
            return 0.0

        target = self.learning_rate * hvi_full
        ref = self._ref_point_transformed
        delta = obj_trans - ref  # direction from reference to objective

        lo, hi = 0.0, 1.0
        for _ in range(30):        # 30 iterations → ~1e-9 precision
            mid = (lo + hi) * 0.5
            disc = mid * delta + ref
            hvi_mid = (
                self._hv_of_trans_front(t_e + [disc])
                - self._hv_of_trans_front(t_e)
            )
            if hvi_mid < target:
                lo = mid
            else:
                hi = mid

        # Return lo (conservative side) so HVI(disc, T_e) ≤ target.
        # Appendix B: "we restrict the error to be in the negative direction"
        # to prevent T_e expanding further than intended.
        return lo

    def _update_threshold_front(self, cell_idx: int, disc_trans: np.ndarray) -> None:
        """Pareto-dominance update of T_e with the discounted objective vector.

        disc_trans is in the transformed (all-minimisation) space.
        Dominated members of T_e are pruned; disc_trans is added if non-dominated.
        """
        t_e = self.threshold_fronts.get(cell_idx, [])

        # Reject if disc_trans is dominated by any existing T_e member
        for v in t_e:
            if self._dominates_min(v, disc_trans):
                return

        # Remove members of T_e that disc_trans dominates
        new_t_e = [v for v in t_e if not self._dominates_min(disc_trans, v)]
        new_t_e.append(disc_trans.copy())
        self.threshold_fronts[cell_idx] = new_t_e

    # ------------------------------------------------------------------
    # Cycle restart (Section 4.3)
    # ------------------------------------------------------------------

    def _check_cycle_restart(self, cell_assignments: np.ndarray) -> None:
        """Restart emitters that have been stuck cycling in the same cell.

        For each emitter, find the modal CVT cell its solutions landed in this
        generation.  If that cell has been the modal cell for more than
        K = n_centroids / n_emitters consecutive generations,
        restart the emitter from a new latent vector.

        This implements Section 4.3 of Zhao & Nikolaidis (GECCO '25).
        """
        # K = (n_centroids / n_emitters): fixed staleness threshold, independent of
        # total decoded solutions (which grows each gen and made K ≈ 1000+).
        # After K consecutive gens in the same modal cell, restart the emitter.
        n_emitters = max(1, len(self.scheduler.emitters))
        K = max(5.0, float(self.n_centroids) / n_emitters)
        if self.generation < 5:
            return  # too early in the run; don't restart yet

        batch_sizes = [e.batch_size for e in self.scheduler.emitters]
        offset = 0
        for emitter_idx, bsize in enumerate(batch_sizes):
            batch_cells = cell_assignments[offset:offset + bsize]
            offset += bsize

            valid = batch_cells[batch_cells >= 0]
            if len(valid) == 0:
                # All decodes failed — count as stuck in current cell so restart triggers
                self._emitter_cell_streak[emitter_idx] += 1
                if self._emitter_cell_streak[emitter_idx] > K:
                    self._restart_emitter(emitter_idx)
                    self._emitter_cell_streak[emitter_idx] = 0
                    self._emitter_last_cell[emitter_idx] = -1
                continue

            modal_cell = int(np.bincount(valid).argmax())

            if modal_cell == self._emitter_last_cell[emitter_idx]:
                self._emitter_cell_streak[emitter_idx] += 1
            else:
                self._emitter_last_cell[emitter_idx] = modal_cell
                self._emitter_cell_streak[emitter_idx] = 1

            if self._emitter_cell_streak[emitter_idx] > K:
                self._restart_emitter(emitter_idx)
                self._emitter_cell_streak[emitter_idx] = 0
                self._emitter_last_cell[emitter_idx] = -1

    def _restart_emitter(self, emitter_idx: int) -> None:
        """Force-restart emitter emitter_idx from a new latent vector.

        Samples new x0 from a random cell of the adaptation archive.
        Falls back to a standard-normal random vector if the archive is empty.
        Resets the emitter's CMA-ES optimizer via the private _opt.reset() API.
        """
        emitter = self.scheduler.emitters[emitter_idx]

        # Pick a new starting point from a *random* cell in the adaptation archive
        # so the emitter moves away from its current exhausted cell.
        new_x0 = None
        try:
            df = self.scheduler.archive.data(return_type='pandas')
            if len(df) > 0:
                row = df.sample(1).iloc[0]
                sol_cols = sorted(
                    [c for c in df.columns if c.startswith('solution_')],
                    key=lambda c: int(c.split('_')[-1]),
                )
                new_x0 = np.array([row[c] for c in sol_cols], dtype=float)
        except Exception:
            pass

        if new_x0 is None:
            new_x0 = np.random.randn(self._embed_dim)

        # Reset the underlying CMA-ES optimizer.
        # pyribs 0.9 exposes this via emitter._opt.reset(x0).
        restarted = False
        try:
            emitter._opt.reset(new_x0)
            restarted = True
        except AttributeError:
            pass

        if restarted:
            print(f"  [Cycle restart] Emitter {emitter_idx} → new x0 "
                  f"(was in cell {self._emitter_last_cell[emitter_idx]})")

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _record_failure(self, smiles: str, props: Dict, generation: int) -> None:
        """Log a failed evaluation (xTB/QC error) without inserting into archive."""
        self.total_eval_failures += 1
        self.failed_evaluations.append({
            'smiles': smiles,
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

    def _extract_objectives(self, props: Dict) -> np.ndarray:
        return np.array([float(props.get(k, 0.0)) for k in self.objective_keys])

    def _extract_measures(self, props: Dict) -> np.ndarray:
        return np.array([float(props.get(k, 0.0)) for k in self.measure_keys])

    # ------------------------------------------------------------------
    # Nearest-neighbour decode
    # ------------------------------------------------------------------

    def _decode_to_smiles(self, z: np.ndarray) -> Optional[str]:
        """Map a CMA-ES latent vector to a SMILES string.

        Strategy:
          1. Find the nearest (embedding, SMILES) pair in `_z_smiles_pool` by
             Euclidean distance.
          2. Apply `mutate_fn` to the nearest SMILES.
          3. If mutation fails, return the nearest SMILES unchanged.
          4. If the pool is empty, call `generate_fn()` as a cold-start fallback.
        """
        if not self._z_smiles_pool:
            return self.generate_fn()

        pool_vecs = np.stack([p[0] for p in self._z_smiles_pool])  # (N, D)
        dists = np.linalg.norm(pool_vecs - z, axis=1)
        nearest_idx = int(np.argmin(dists))
        nearest_smiles = self._z_smiles_pool[nearest_idx][1]

        mutated = self.mutate_fn(nearest_smiles)
        return mutated if mutated is not None else nearest_smiles

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Seed both archives with random molecules encoded into the ChemBERTa-UMAP space."""
        print(f"Initialising MO-CMA-MAE archive with {self.random_init_size} molecules...")
        n_added = 0

        for i in range(self.random_init_size):
            smiles = self.generate_fn()
            if smiles is None:
                continue

            try:
                z = self.embedder.embed([smiles])[0].astype(np.float64)
            except Exception:
                z = np.random.randn(self._embed_dim).astype(np.float64)

            # Add to pool so _decode_to_smiles has neighbours from the start
            self._z_smiles_pool.append((z, smiles))

            props = self._evaluate_with_cache(smiles)
            if props.get('error') is not None:
                self._record_failure(smiles, props, generation=0)
                continue

            obj_vec = self._extract_objectives(props)
            meas = self._extract_measures(props)

            # T_e-based HVI (same logic as step())
            cell_idx = self._get_cell_idx(meas)
            obj_trans = self._transform_objectives(obj_vec)
            t_e = self.threshold_fronts.get(cell_idx, [])
            hvi = (
                self._hv_of_trans_front(t_e + [obj_trans])
                - self._hv_of_trans_front(t_e)
            )
            hv_contrib = max(hvi, self.threshold_min)

            # Update T_e and P_e — only when HVI(T_e) > 0 (Algorithm 1 line 11)
            if hvi > 1e-12:
                d_i = self._bisect_discount(obj_trans, t_e)
                if d_i > 1e-9:
                    disc = d_i * (obj_trans - self._ref_point_transformed) + self._ref_point_transformed
                    self._update_threshold_front(cell_idx, disc)
                # P_e: force_add (acceptance based on T_e, not F_e)
                self.result_archive.force_add(smiles, props, niche_idx=cell_idx)
            self.update_molecule_database(smiles, props, z, generation=0)

            # Seed adaptation archive
            try:
                self.scheduler.archive.add_single(z, hv_contrib, meas)
            except Exception:
                pass

            n_added += 1
            if (i + 1) % max(1, self.random_init_size // 5) == 0:
                cov = self.result_archive.get_coverage()
                print(f"  {i+1}/{self.random_init_size} — coverage: {cov:.2%}")

        print(f"Initialisation complete. {n_added} molecules evaluated.")

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self) -> Dict[str, Any]:
        """
        One MO-CMA-MAE generation (Algorithm 1 from the paper):
          1. Ask pyribs for latent vectors (CMA-ES sampling)
          2. Decode z → SMILES via nearest-neighbour + mutation
          3. Evaluate each molecule
          4. Compute HVI(f, T_e) as scalar CMA-ES signal
          5. If HVI > 0: bisect for d_i, insert d_i*(f-ref)+ref into T_e
          6. Tell pyribs (updates thresholds + CMA-ES covariance matrices)
          7. Update result archive F_e (CVTMOMEArchive, Pareto dominance)
        """
        import time as _time
        _t_step_start = _time.monotonic()

        solutions = self.scheduler.ask()   # (total_batch, latent_dim)
        _t_after_ask = _time.monotonic()
        n = len(solutions)

        hv_contribs = np.full(n, self.threshold_min, dtype=float)
        measures_batch = np.zeros((n, len(self.measure_keys)))
        # cell_assignments[i] = CVT cell index of solution i, or -1 for failed decode.
        # Used by cycle-restart tracking after tell().
        cell_assignments = np.full(n, -1, dtype=int)
        n_decoded = 0

        # ── Decode all candidates (nearest-neighbour + mutation) ──────────
        decoded = []
        for z in solutions:
            decoded.append(self._decode_to_smiles(z))
        _t_after_decode = _time.monotonic()

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

            obj_vec = self._extract_objectives(props)
            meas = self._extract_measures(props)

            # Look up this solution's CVT cell and its threshold front T_e
            cell_idx = self._get_cell_idx(meas)
            obj_trans = self._transform_objectives(obj_vec)  # 1-D, all-min space
            t_e = self.threshold_fronts.get(cell_idx, [])

            # HVI relative to T_e  — the paper's improvement signal
            hvi = (
                self._hv_of_trans_front(t_e + [obj_trans])
                - self._hv_of_trans_front(t_e)
            )
            hv_contribs[i] = max(hvi, self.threshold_min)
            measures_batch[i] = meas
            cell_assignments[i] = cell_idx
            self._cell_visit_counts[cell_idx] = self._cell_visit_counts.get(cell_idx, 0) + 1
            n_decoded += 1

            # Threshold front update + real front update — only when HVI(T_e) > 0
            # Algorithm 1, line 11: "if Φ_i > 0 then"
            if hvi > 1e-12:
                # T_e update: bisection finds d_i; discounted vector inserted into T_e
                d_i = self._bisect_discount(obj_trans, t_e)
                if d_i > 1e-9:
                    disc = (
                        d_i * (obj_trans - self._ref_point_transformed)
                        + self._ref_point_transformed
                    )
                    self._update_threshold_front(cell_idx, disc)

                # P_e update: Algorithm 1 line 12 — add unconditionally, remove dominated.
                # Acceptance is based on T_e dominance, not F_e dominance.
                self.result_archive.force_add(smiles, props, niche_idx=cell_idx)
            self.update_molecule_database(smiles, props, z, self.generation + 1)
            # Grow the decode pool with every evaluated molecule so nearest-
            # neighbour decode improves coverage over time.
            self._z_smiles_pool.append((z.astype(np.float64), smiles))

        _t_after_eval = _time.monotonic()

        # Feed HVI(f, T_e) scalars back to CMA-ES (drives covariance update)
        self.scheduler.tell(hv_contribs, measures_batch)
        _t_after_tell = _time.monotonic()
        self.generation += 1

        # Cycle restart (Section 4.3): restart emitters stuck in exhausted cells.
        # Called after tell() so CMA-ES state is in a consistent resting state.
        self._check_cycle_restart(cell_assignments)
        _t_after_restart = _time.monotonic()

        decode_rate = n_decoded / n if n > 0 else 0.0
        valid = hv_contribs[hv_contribs > self.threshold_min]
        fail_rate = (
            self.total_eval_failures / self.total_evaluations
            if self.total_evaluations > 0 else 0.0
        )

        # Diversity metrics — expensive O(n) scaffold computation, throttled
        _t_before_div = _time.monotonic()
        if self.generation % self.diversity_log_interval == 0:
            archive_smiles = [e['solution'] for e in self.result_archive.get_all_solutions()]
            div = compute_diversity_metrics(archive_smiles, max_sample=500)
            self._cached_diversity = div
        else:
            div = self._cached_diversity
        _t_after_div = _time.monotonic()

        # Emit timing every 50 gens when step_timing=True (for diagnosing bottlenecks)
        if self.step_timing and self.generation % 50 == 0:
            _t_total = _t_after_div - _t_step_start
            print(
                f"  [TIMING gen {self.generation}] "
                f"ask={_t_after_ask-_t_step_start:.2f}s "
                f"decode={_t_after_decode-_t_after_ask:.2f}s "
                f"eval={_t_after_eval-_t_after_decode:.2f}s "
                f"tell={_t_after_tell-_t_after_eval:.2f}s "
                f"restart={_t_after_restart-_t_after_tell:.2f}s "
                f"div={_t_after_div-_t_before_div:.2f}s "
                f"total={_t_total:.2f}s"
            )

        return {
            'generation': self.generation,
            'coverage': self.result_archive.get_coverage(),
            'archive_size': self.result_archive.n_filled,
            'moqd_score': self.result_archive.compute_moqd_score(),
            'global_hypervolume': self.result_archive.compute_global_hypervolume(),
            'max_hv_contrib': float(valid.max()) if len(valid) else 0.0,
            'mean_hv_contrib': float(valid.mean()) if len(valid) else 0.0,
            'decode_success_rate': decode_rate,
            'total_evaluations': self.total_evaluations,
            'int_div': div['int_div'],
            'scaffold_count': div['scaffold_count'],
            'n_unique': div['n_unique'],
            'eval_fail_count': self.total_eval_failures,
            'eval_fail_rate': fail_rate,
        }

    # ------------------------------------------------------------------
    # Run loop
    # ------------------------------------------------------------------

    def _log_stats(self, stats: Dict):
        with open(self._stats_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                stats['generation'], stats['coverage'], stats['archive_size'],
                stats['moqd_score'], stats['global_hypervolume'],
                stats['max_hv_contrib'], stats['mean_hv_contrib'],
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
        """Main MO-CMA-MAE optimisation loop."""
        if self.result_archive.n_filled == 0:
            self.initialize()

        print(f"\nRunning MO-CMA-MAE for {n_generations} generations...")
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
                    f"MOQD={stats['moqd_score']:10.3f}, "
                    f"GHV={stats['global_hypervolume']:10.3f}, "
                    f"IntDiv={stats['int_div']:.3f}, "
                    f"Scaffolds={stats['scaffold_count']:4d}, "
                    f"Decode={stats['decode_success_rate']:5.1%}"
                )

        print("\nMO-CMA-MAE complete!")
        print(f"Final coverage:      {self.result_archive.get_coverage():.2%}")
        print(f"Final archive size:  {self.result_archive.n_filled}")
        print(f"Final MOQD score:    {self.result_archive.compute_moqd_score():.3f}")
        print(f"Final global HV:     {self.result_archive.compute_global_hypervolume():.3f}")
        print(f"Total evaluations:   {self.total_evaluations}")
        if self.total_eval_failures > 0:
            fail_rate = self.total_eval_failures / self.total_evaluations
            print(f"Eval failures:       {self.total_eval_failures} ({fail_rate:.1%})")

        self.save_archive(n_generations)
        self.save_molecule_database()
        return history

    # ------------------------------------------------------------------
    # Database & archive persistence
    # ------------------------------------------------------------------

    def update_molecule_database(
        self, smiles: str, props: Dict, z: np.ndarray, generation: int
    ):
        existing = next((m for m in self.all_molecules if m['smiles'] == smiles), None)
        if existing:
            if generation < existing['generation']:
                existing['generation'] = generation
        else:
            entry: Dict = {
                'smiles': smiles,
                'generation': generation,
                'z_vector': z.tolist(),
            }
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

        if self.failed_evaluations:
            fail_file = self.output_dir / 'failed_evaluations.json'
            with open(fail_file, 'w') as f:
                json.dump(self.failed_evaluations, f, indent=2)
            print(f"Saved {len(self.failed_evaluations)} failed evaluations to {fail_file}")

    def save_archive(self, generation: int):
        """Save current result_archive Pareto fronts to JSON."""
        archive_data = {'generation': generation, 'cells': []}

        for (cell_idx,), front in self.result_archive.iter_filled_cells():
            cell_data = {
                'cell_index': int(cell_idx),
                'front_size': len(front),
                'solutions': [],
            }
            for entry in front:
                cell_data['solutions'].append({
                    'smiles': entry['solution'],
                    'objectives': {k: float(v)
                                   for k, v in zip(self.objective_keys,
                                                   entry['objectives'])},
                    'properties': {k: v for k, v in entry['properties'].items()
                                   if isinstance(v, (int, float, str, type(None)))},
                })
            archive_data['cells'].append(cell_data)

        filename = self.output_dir / f'archive_gen_{generation:04d}.json'
        with open(filename, 'w') as f:
            json.dump(archive_data, f, indent=2)
        print(f"Saved archive ({len(archive_data['cells'])} cells) to {filename}")

    def get_best_solutions(self, n: int = 10) -> List[Dict]:
        """Return top-n solutions by first maximised objective from the global Pareto front."""
        global_front = self.result_archive.get_global_pareto_front()
        if not global_front:
            return []
        first_max_key = next(
            (k for k, (opt, _) in zip(self.objective_keys, self.optimize_objectives)
             if opt == 'max'),
            self.objective_keys[0],
        )
        return sorted(
            global_front,
            key=lambda e: e['properties'].get(first_max_key, 0.0),
            reverse=True,
        )[:n]

    def get_statistics(self) -> Dict[str, Any]:
        return {
            'generation': self.generation,
            'coverage': self.result_archive.get_coverage(),
            'archive_size': self.result_archive.n_filled,
            'moqd_score': self.result_archive.compute_moqd_score(),
            'global_hypervolume': self.result_archive.compute_global_hypervolume(),
            'total_evaluations': self.total_evaluations,
        }

    @staticmethod
    def recalculate_from_database(results_dir: str, archive_config: Dict = None):
        """
        Rebuild MOQD / global HV metrics from an existing all_molecules_database.json
        without re-running QC evaluations.
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
                'objective_keys': ['beta_gamma_ratio', 'total_energy_atom_ratio',
                                   'alpha_range_distance', 'homo_lumo_gap_range_distance'],
                'optimize_objectives': [('max', None), ('min', None), ('min', None), ('min', None)],
                'reference_point': [0.0, 0.0, 500.0, 100.0],
                'measure_keys': ['num_atoms', 'num_bonds'],
                'measure_ranges': [(1, 50), (0, 60)],
            }

        obj_keys = archive_config['objective_keys']
        opt_objs = archive_config.get('optimize_objectives',
                                      [('max', None)] * len(obj_keys))
        measure_keys = archive_config['measure_keys']

        from pymoo.indicators.hv import Hypervolume

        def make_ref():
            return np.array([0.0 if opt == 'max' else 1000.0 for opt, _ in opt_objs])

        def transform(objs):
            t = objs.copy()
            for i, (opt, _) in enumerate(opt_objs):
                if opt == 'max':
                    t[:, i] = -t[:, i]
            return t

        hv_ind = Hypervolume(ref_point=make_ref())

        def cell_hv(front_objs):
            if not front_objs:
                return 0.0
            arr = np.array(front_objs)
            try:
                return float(hv_ind(transform(arr)))
            except Exception:
                return 0.0

        recalc_dir = results_path / 'recalculated'
        recalc_dir.mkdir(exist_ok=True)

        max_gen = max(m.get('generation', 0) for m in molecules)
        stats_rows = []

        for gen in range(max_gen + 1):
            cells: Dict[str, list] = {}
            for mol in molecules:
                if mol.get('generation', 0) > gen:
                    continue
                objs = [mol.get(k) for k in obj_keys]
                if any(o is None for o in objs):
                    continue
                meas_key = str([mol.get(k, 0) for k in measure_keys])
                if meas_key not in cells:
                    cells[meas_key] = []
                cells[meas_key].append([float(o) for o in objs])

            moqd = sum(cell_hv(front) for front in cells.values())
            all_objs = [o for front in cells.values() for o in front]
            ghv = cell_hv(all_objs) if all_objs else 0.0

            stats_rows.append({
                'generation': gen,
                'coverage': len(cells),
                'moqd_score': moqd,
                'global_hypervolume': ghv,
            })
            if gen % 10 == 0 or gen == max_gen:
                print(f"  Gen {gen}: cells={len(cells)}, MOQD={moqd:.3f}, GHV={ghv:.3f}")

        out_file = recalc_dir / 'recalculated_stats.json'
        with open(out_file, 'w') as f:
            json.dump(stats_rows, f, indent=2)
        print(f"Recalculation complete. Stats saved to {out_file}")
