"""
Rastrigin QD Benchmark: validates DBSA as a general surrogate-QD method.

Compares three MAP-Elites conditions under a limited oracle budget:
  1. MAP-Elites (no surrogate) — all candidates evaluated
  2. MAP-Elites + SA (quality-only surrogate)
  3. MAP-Elites + DBSA (quality + behavioural novelty surrogate)

Task: 10D Rastrigin with 2D behavioural descriptor (x_0, x_1).
CVT: 50 cells in [-5.12, 5.12]^2 behavioral space.
Oracle budget: 2,000 evaluations total.
CMA-ES generates 100 candidates/generation; surrogate screens to 20 for oracle eval.

Usage:
    python rastrigin_qd_benchmark.py [--seeds 10] [--budget 2000] [--output results/rastrigin]
"""

from __future__ import annotations
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from scipy.cluster.vq import kmeans2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ─────────────────────────────────────────────────────────────────────────────
# Rastrigin oracle and behavioral descriptor
# ─────────────────────────────────────────────────────────────────────────────

DOMAIN_LOW  = -5.12
DOMAIN_HIGH =  5.12
N_DIM       = 10
A           = 10.0


def rastrigin(x: np.ndarray) -> float:
    """Negated Rastrigin (higher = better). x: shape (N_DIM,)"""
    return float(-(A * N_DIM + np.sum(x**2 - A * np.cos(2 * np.pi * x))))


def behavior(x: np.ndarray) -> np.ndarray:
    """2D behavioral descriptor: first two dimensions. x: (N_DIM,) → (2,)"""
    return x[:2].copy()


# ─────────────────────────────────────────────────────────────────────────────
# CVT archive
# ─────────────────────────────────────────────────────────────────────────────

class CVTArchive:
    """Minimal CVT archive for the Rastrigin benchmark."""

    def __init__(self, n_cells: int, low: float, high: float, seed: int = 42):
        rng = np.random.RandomState(seed)
        # Seed centroids from uniform samples in behavioral space
        seed_pts = rng.uniform(low, high, size=(5000, 2)).astype(np.float32)
        self.centroids, _ = kmeans2(seed_pts, n_cells, seed=seed,
                                    minit='points', missing='warn', iter=100)
        self.n_cells = n_cells
        self._solutions: dict[int, tuple[np.ndarray, float]] = {}  # cell → (x, quality)

    def cell_of(self, beh: np.ndarray) -> int:
        diffs = self.centroids - beh[None, :]
        return int(np.argmin((diffs**2).sum(axis=1)))

    def try_add(self, x: np.ndarray, quality: float, beh: np.ndarray) -> bool:
        k = self.cell_of(beh)
        if k not in self._solutions or self._solutions[k][1] < quality:
            self._solutions[k] = (x.copy(), quality)
            return True
        return False

    @property
    def coverage(self) -> float:
        return len(self._solutions) / self.n_cells

    @property
    def mean_quality(self) -> float:
        if not self._solutions:
            return float('-inf')
        return float(np.mean([v[1] for v in self._solutions.values()]))

    @property
    def occupancy(self) -> np.ndarray:
        """Returns (n_cells,) array of per-cell solution count (0 or 1)."""
        occ = np.zeros(self.n_cells, dtype=np.int32)
        for k in self._solutions:
            occ[k] = 1
        return occ

    def sample_elites(self, n: int, rng: np.random.RandomState) -> list[np.ndarray]:
        """Sample n solutions from filled cells for mutation."""
        filled = list(self._solutions.values())
        if not filled:
            return [rng.uniform(DOMAIN_LOW, DOMAIN_HIGH, N_DIM) for _ in range(n)]
        idxs = rng.choice(len(filled), size=n, replace=True)
        return [filled[i][0].copy() for i in idxs]


# ─────────────────────────────────────────────────────────────────────────────
# Simple quality + behavior surrogate (2-head MLP, matches DBSA design)
# ─────────────────────────────────────────────────────────────────────────────

class _SurrogateNet(nn.Module):
    def __init__(self, input_dim: int, behavior_dim: int, dropout: float = 0.1):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64),        nn.ReLU(), nn.Dropout(dropout),
        )
        self.head_quality = nn.Linear(64, 1)
        self.head_behavior = nn.Linear(64, behavior_dim)

    def forward(self, x):
        h = self.trunk(x)
        return self.head_quality(h).squeeze(-1), self.head_behavior(h)


class Surrogate:
    """Dual-head surrogate for the Rastrigin benchmark."""

    def __init__(self, input_dim: int, behavior_dim: int, n_cells: int,
                 centroids: np.ndarray, mode: str = 'dbsa', device: str = 'cpu'):
        assert mode in ('quality', 'dbsa')
        self.mode = mode
        self.input_dim = input_dim
        self.behavior_dim = behavior_dim
        self.n_cells = n_cells
        self.centroids = centroids.astype(np.float32)
        self.device = device
        self._model: _SurrogateNet | None = None
        self._tau: float | None = None
        self._obs_x: list[np.ndarray] = []
        self._obs_q: list[float] = []
        self._obs_b: list[np.ndarray] = []

    def add(self, x: np.ndarray, quality: float, beh: np.ndarray):
        self._obs_x.append(x.astype(np.float32))
        self._obs_q.append(float(quality))
        self._obs_b.append(beh.astype(np.float32))

    def refit(self, epochs: int = 50, batch_size: int = 32, lr: float = 1e-3):
        n = len(self._obs_x)
        if n < 10:
            return
        X = torch.tensor(np.stack(self._obs_x), dtype=torch.float32)
        Q = torch.tensor(self._obs_q, dtype=torch.float32)
        B = torch.tensor(np.stack(self._obs_b), dtype=torch.float32)
        loader = DataLoader(TensorDataset(X, Q, B), batch_size=batch_size,
                            shuffle=True, drop_last=False)
        if self._model is None:
            self._model = _SurrogateNet(self.input_dim, self.behavior_dim).to(self.device)
        opt = torch.optim.Adam(self._model.parameters(), lr=lr)
        self._model.train()
        for _ in range(epochs):
            for xb, qb, bb in loader:
                xb, qb, bb = xb.to(self.device), qb.to(self.device), bb.to(self.device)
                q_pred, b_pred = self._model(xb)
                loss = nn.functional.mse_loss(q_pred, qb)
                if self.mode == 'dbsa':
                    loss = loss + 0.1 * nn.functional.mse_loss(b_pred, bb)
                opt.zero_grad(); loss.backward(); opt.step()
        self._model.eval()
        # Compute tau for soft assignment
        c = self.centroids
        diffs = c[:, None, :] - c[None, :, :]
        self._tau = float((diffs**2).sum(-1).mean()) / 2.0 + 1e-8

    def score(self, candidates: np.ndarray, occupancy: np.ndarray,
              beta: float = 2.0, gamma: float = 1.0,
              T_mc: int = 20) -> np.ndarray:
        """Score candidates for acquisition. Returns (N,) scores."""
        if self._model is None or len(self._obs_x) < 10:
            rng = np.random.RandomState()
            return rng.rand(len(candidates))

        X = torch.tensor(candidates.astype(np.float32), device=self.device)
        # MC-dropout for quality uncertainty
        self._model.train()  # dropout active
        q_samples = []
        with torch.no_grad():
            for _ in range(T_mc):
                q_pred, b_pred = self._model(X)
                q_samples.append(q_pred.cpu().numpy())
        self._model.eval()
        q_arr = np.stack(q_samples, axis=0)  # (T, N)
        mu_q = q_arr.mean(0)
        std_q = q_arr.std(0) + 1e-8
        ucb = mu_q + beta * std_q

        if self.mode == 'quality':
            return ucb

        # DBSA: behavioural novelty from regression head
        with torch.no_grad():
            _, b_pred = self._model(X)
        b_np = b_pred.cpu().numpy()  # (N, behavior_dim)

        cents = self.centroids  # (K, 2)
        diffs = b_np[:, None, :] - cents[None, :, :]
        dists_sq = (diffs**2).sum(-1)  # (N, K)
        logits = -dists_sq / self._tau
        logits -= logits.max(1, keepdims=True)
        w = np.exp(logits); w /= w.sum(1, keepdims=True)
        novelty = (w / (1.0 + occupancy[None, :])).sum(1)

        return ucb * (1.0 + gamma * novelty)


# ─────────────────────────────────────────────────────────────────────────────
# MAP-Elites runner
# ─────────────────────────────────────────────────────────────────────────────

def run_condition(condition: str, oracle_budget: int, seed: int,
                  n_cells: int = 50, batch_gen: int = 100,
                  screen_fraction: float = 0.20) -> dict:
    """
    Run one condition (no_surrogate / quality / dbsa) for oracle_budget evaluations.

    Returns dict with coverage_curve, quality_curve, n_evals lists.
    """
    assert condition in ('no_surrogate', 'quality', 'dbsa')
    rng = np.random.RandomState(seed)

    archive = CVTArchive(n_cells=n_cells, low=DOMAIN_LOW, high=DOMAIN_HIGH, seed=42)
    n_oracle_screen = max(1, int(batch_gen * screen_fraction))
    surrogate = None
    if condition in ('quality', 'dbsa'):
        surrogate = Surrogate(
            input_dim=N_DIM, behavior_dim=2, n_cells=n_cells,
            centroids=archive.centroids, mode=condition
        )

    coverage_curve, quality_curve, eval_curve = [], [], []
    total_evals = 0

    # Random initialisation: 50 solutions
    init_size = min(50, oracle_budget // 4)
    for _ in range(init_size):
        x = rng.uniform(DOMAIN_LOW, DOMAIN_HIGH, N_DIM)
        q = rastrigin(x); b = behavior(x)
        archive.try_add(x, q, b)
        if surrogate is not None:
            surrogate.add(x, q, b)
        total_evals += 1

    if surrogate is not None:
        surrogate.refit()

    retrain_interval = 10  # generations
    gen = 0

    while total_evals < oracle_budget:
        # Generate candidates via Gaussian perturbation of archive elites
        parents = archive.sample_elites(batch_gen, rng)
        sigma = 0.5
        candidates = np.stack([
            np.clip(p + rng.randn(N_DIM) * sigma, DOMAIN_LOW, DOMAIN_HIGH)
            for p in parents
        ])

        if condition == 'no_surrogate':
            to_eval = candidates
        else:
            # Screen with surrogate
            occ = archive.occupancy
            scores = surrogate.score(candidates, occ)
            top_k = int(np.ceil(len(candidates) * screen_fraction))
            top_k = min(top_k, oracle_budget - total_evals)
            top_idx = np.argsort(scores)[::-1][:top_k]
            to_eval = candidates[top_idx]

        # Evaluate with oracle
        to_eval = to_eval[:oracle_budget - total_evals]
        for x in to_eval:
            q = rastrigin(x); b = behavior(x)
            archive.try_add(x, q, b)
            if surrogate is not None:
                surrogate.add(x, q, b)
            total_evals += 1

        # Periodic refit
        if surrogate is not None and gen % retrain_interval == 0 and total_evals >= 20:
            surrogate.refit()

        coverage_curve.append(archive.coverage)
        quality_curve.append(archive.mean_quality)
        eval_curve.append(total_evals)
        gen += 1

    return {
        'condition': condition,
        'seed': seed,
        'coverage_curve': coverage_curve,
        'quality_curve': quality_curve,
        'eval_curve': eval_curve,
        'final_coverage': archive.coverage,
        'final_quality': archive.mean_quality,
        'total_evals': total_evals,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, default=10)
    parser.add_argument('--budget', type=int, default=2000)
    parser.add_argument('--n-cells', type=int, default=50)
    parser.add_argument('--output', type=str, default='results/rastrigin')
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_list = [1, 33, 42, 55, 88, 2, 7, 13, 21, 99][:args.seeds]
    conditions = ['no_surrogate', 'quality', 'dbsa']

    all_results = {c: [] for c in conditions}

    for condition in conditions:
        print(f'\nCondition: {condition}')
        for seed in seed_list:
            t0 = time.time()
            r = run_condition(
                condition=condition,
                oracle_budget=args.budget,
                seed=seed,
                n_cells=args.n_cells,
            )
            elapsed = time.time() - t0
            print(f'  seed={seed}: coverage={r["final_coverage"]:.3f}  '
                  f'quality={r["final_quality"]:.2f}  ({elapsed:.1f}s)')
            all_results[condition].append(r)

    # Summary statistics
    print('\n=== Summary ===')
    print(f'{"Condition":<15}  {"Coverage":>10}  {"Quality":>10}')
    for cond, runs in all_results.items():
        covs = [r['final_coverage'] for r in runs]
        quals = [r['final_quality'] for r in runs]
        print(f'{cond:<15}  '
              f'{np.mean(covs):.3f}±{np.std(covs):.3f}  '
              f'{np.mean(quals):.2f}±{np.std(quals):.2f}')

    # Save results
    out_file = out_dir / f'rastrigin_budget{args.budget}_seeds{args.seeds}.json'
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nResults saved to {out_file}')


if __name__ == '__main__':
    main()
