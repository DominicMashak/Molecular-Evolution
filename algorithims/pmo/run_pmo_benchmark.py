"""
PMO Benchmark Runner
====================
Run any optimizer (MOME, MAP-Elites, MO-CMA-MAE, CMA-MAE) against a PMO oracle
and report PMO-standard AUC metrics.

The benchmark wraps the oracle's budget so the run terminates automatically
once the evaluation budget is exhausted.

Usage examples
--------------
# QED single-objective (MAP-Elites)
python run_pmo_benchmark.py --oracle qed --algorithm map_elites \\
    --budget 10000 --seed 42 --output_dir results/qed_map_elites_42

# Penalized logP (MOME, 2 objectives: penalized_logp + QED)
python run_pmo_benchmark.py --oracle penalized_logp --algorithm mome \\
    --budget 10000 --seed 42 --output_dir results/plogp_mome_42

# QED (MO-CMA-MAE, objectives: qed + sa)
python run_pmo_benchmark.py --oracle qed --algorithm mo_cma_mae \\
    --budget 10000 --seed 42 --output_dir results/qed_mo_cma_mae_42

# Compare all algorithms on QED (3 seeds each)
for alg in map_elites mome mo_cma_mae; do
  for seed in 1 42 99; do
    python run_pmo_benchmark.py --oracle qed --algorithm $alg \\
      --budget 10000 --seed $seed \\
      --output_dir results/qed_${alg}_${seed}
  done
done
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

# ── path setup ───────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / 'molev_utils'))
sys.path.insert(0, str(_REPO_ROOT / 'algorithims' / 'mome'))
sys.path.insert(0, str(_REPO_ROOT / 'algorithims' / 'map_elites'))

from drug.pmo_oracle import PMOOracle, BudgetExhaustedError, create_oracle
from molecule_generator import MoleculeGenerator


def _build_embedder(args, generator):
    """Fit a MolecularEmbedder (ChemBERTa-2 + UMAP) on random molecules."""
    from molecular_embedder import MolecularEmbedder
    n = getattr(args, 'embedding_sample_size', 5000)
    dims = getattr(args, 'embedding_dims', 10)
    device = getattr(args, 'embedding_device', 'auto')
    model = getattr(args, 'embedding_model', 'DeepChem/ChemBERTa-77M-MTR')
    print(f"Fitting ChemBERTa embedder on {n} molecules (UMAP → {dims}D)...")
    raw = generator.generate_initial_population(n)
    smiles = [generator.decode_to_smiles(s) for s in raw]
    smiles = [s for s in smiles if s]
    embedder = MolecularEmbedder(
        model_name=model,
        n_components=dims,
        device=device,
        random_state=args.seed,
    )
    embedder.fit(smiles)
    print(f"Embedder fitted. {dims}D UMAP space ready.")
    return embedder


def _make_generate_fn(generator: MoleculeGenerator):
    """Return a generate_fn that produces one random molecule at a time."""
    _pool: list = []

    def generate_fn():
        nonlocal _pool
        if not _pool:
            _pool = generator.generate_initial_population(50)
        if _pool:
            return _pool.pop()
        return None

    return generate_fn


# ── helpers ──────────────────────────────────────────────────────────────────

def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        from rdkit import rdBase
        rdBase.SeedRandomNumberGenerator(seed)
    except Exception:
        pass


def _make_evaluate_fn(oracle: PMOOracle):
    """Return a safe wrapper that returns a zero dict instead of raising."""
    def evaluate(smiles: str) -> dict:
        try:
            return oracle.evaluate(smiles)
        except BudgetExhaustedError:
            raise  # let optimizers catch this
    return evaluate


# ── optimizer builders ────────────────────────────────────────────────────────

def _add_atom_measures(evaluate_fn):
    """Wrap evaluate_fn to add integer num_atoms and num_bonds for grid archives."""
    def wrapped(smiles: str) -> dict:
        props = evaluate_fn(smiles)
        if smiles:
            try:
                from rdkit import Chem
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    props['num_atoms'] = int(mol.GetNumAtoms())
                    props['num_bonds'] = int(mol.GetNumBonds())
            except Exception:
                pass
        if 'num_atoms' not in props:
            props['num_atoms'] = 0
        if 'num_bonds' not in props:
            props['num_bonds'] = 0
        return props
    return wrapped


def _run_map_elites(args, oracle: PMOOracle, generator: MoleculeGenerator,
                    evaluate_fn) -> None:
    """Run MAP-Elites (single-objective) against a PMO oracle."""
    sys.path.insert(0, str(_REPO_ROOT / 'algorithims' / 'map_elites'))
    import archive as ma
    import optimizer as mo

    # Grid archive: num_atoms (0–79) × num_bonds (0–89)
    archive = ma.MAPElitesArchive(
        measure_dims=[80, 90],
        measure_keys=['num_atoms', 'num_bonds'],
        objective_key=args.oracle,
    )

    eval_fn = _add_atom_measures(evaluate_fn)

    opt = mo.MAPElitesOptimizer(
        archive=archive,
        generate_fn=_make_generate_fn(generator),
        mutate_fn=lambda s: generator.mutate(s),
        evaluate_fn=eval_fn,
        random_init_size=min(args.init_size, args.budget // 5),
        output_dir=args.output_dir,
    )

    try:
        opt.run(
            n_generations=args.budget,
            log_frequency=args.log_frequency,
            save_frequency=args.save_frequency,
        )
    except BudgetExhaustedError:
        pass  # clean exit when budget exhausted


def _run_mome(args, oracle: PMOOracle, generator: MoleculeGenerator,
              evaluate_fn) -> None:
    """Run MOME (multi-objective) against a PMO oracle."""
    sys.path.insert(0, str(_REPO_ROOT / 'algorithims' / 'mome'))
    import archive as ma
    import optimizer as mo

    # Objectives: primary oracle + QED/SA as secondary
    if args.oracle == 'qed':
        objective_keys = ['qed', 'sa']
        optimize_objectives = [('max', None), ('max', None)]
        reference_point = [0.0, 0.0]
    elif args.oracle == 'penalized_logp':
        objective_keys = ['penalized_logp', 'qed']
        optimize_objectives = [('max', None), ('max', None)]
        reference_point = [-20.0, 0.0]
    elif args.oracle == 'sa':
        objective_keys = ['sa', 'qed']
        optimize_objectives = [('max', None), ('max', None)]
        reference_point = [0.0, 0.0]
    else:
        objective_keys = [args.oracle, 'qed']
        optimize_objectives = [('max', None), ('max', None)]
        reference_point = [0.0, 0.0]

    eval_fn = _add_atom_measures(evaluate_fn)

    if args.archive_type == 'cvt':
        from cvt_archive import CVTMOMEArchive
        archive = CVTMOMEArchive(
            n_centroids=args.n_centroids,
            measure_keys=['num_atoms', 'num_bonds'],
            measure_bounds=[(1, 80), (0, 90)],
            optimize_objectives=optimize_objectives,
            objective_keys=objective_keys,
            n_samples=20000,
            seed=args.seed,
        )
    else:
        # Grid archive: num_atoms (0–79) × num_bonds (0–89)
        archive = ma.MOMEArchive(
            measure_dims=[80, 90],
            measure_keys=['num_atoms', 'num_bonds'],
            objective_keys=objective_keys,
            optimize_objectives=optimize_objectives,
        )

    opt = mo.MOMEOptimizer(
        archive=archive,
        generate_fn=_make_generate_fn(generator),
        mutate_fn=lambda s: generator.mutate(s),
        evaluate_fn=eval_fn,
        random_init_size=min(args.init_size, args.budget // 5),
        output_dir=args.output_dir,
        reference_point=reference_point,
    )

    try:
        opt.run(
            n_generations=args.budget,
            log_frequency=args.log_frequency,
            save_frequency=args.save_frequency,
        )
    except BudgetExhaustedError:
        pass


def _run_mo_cma_mae(args, oracle: PMOOracle, generator: MoleculeGenerator,
                    evaluate_fn) -> None:
    """Run MO-CMA-MAE against a PMO oracle (ChemBERTa-UMAP search space)."""
    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        'mo_cma_mae_optimizer',
        str(_REPO_ROOT / 'algorithims' / 'mo_cma_mae' / 'optimizer.py'),
    )
    op = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(op)
    sys.path.insert(0, str(_REPO_ROOT / 'algorithims' / 'mome'))

    from scipy.cluster.vq import kmeans2
    from ribs.archives import CVTArchive
    from ribs.emitters import EvolutionStrategyEmitter
    from ribs.schedulers import Scheduler
    from cvt_archive import CVTMOMEArchive

    # Objective config: primary oracle + QED as diversity pressure
    _ORACLE_CONFIGS = {
        'penalized_logp': (['penalized_logp', 'qed'], [('max', None), ('max', None)], np.array([-20.0, 0.0])),
        'sa':             (['sa', 'qed'],              [('max', None), ('max', None)], np.array([0.0, 0.0])),
    }
    if args.oracle in _ORACLE_CONFIGS:
        objective_keys, optimize_objectives, reference_point = _ORACLE_CONFIGS[args.oracle]
    else:
        objective_keys = [args.oracle, 'qed']
        optimize_objectives = [('max', None), ('max', None)]
        reference_point = np.array([0.0, 0.0])

    # Fit embedder and determine measures / centroids
    embedder = _build_embedder(args, generator)
    measure_keys = embedder.get_measure_keys()
    measure_bounds = embedder.get_measure_bounds()
    cvt_seed_data = embedder.get_fitted_embeddings()
    embed_dim = embedder.n_components

    print(f"Computing CVT centroids ({args.n_centroids} cells from embedding data)...")
    centroids, _ = kmeans2(cvt_seed_data, args.n_centroids, seed=args.seed,
                           minit='points', iter=100)

    cma_archive = CVTArchive(
        solution_dim=embed_dim,
        centroids=centroids,
        ranges=measure_bounds,
        learning_rate=1.0,
        threshold_min=0.0,
        seed=args.seed,
    )
    result_archive = CVTMOMEArchive(
        n_centroids=args.n_centroids,
        measure_keys=measure_keys,
        measure_bounds=measure_bounds,
        optimize_objectives=optimize_objectives,
        objective_keys=objective_keys,
        precomputed_centroids=centroids,
        reference_point=reference_point.tolist(),
    )

    emitters = [
        EvolutionStrategyEmitter(
            archive=cma_archive,
            x0=np.zeros(embed_dim),
            sigma0=args.sigma0,
            batch_size=args.cma_batch_size,
            seed=args.seed + i,
            ranker='imp',
        )
        for i in range(args.n_emitters)
    ]
    scheduler = Scheduler(cma_archive, emitters)

    def eval_fn(smiles):
        props = evaluate_fn(smiles)
        if smiles:
            try:
                from rdkit import Chem as _Chem
                mol = _Chem.MolFromSmiles(smiles)
                if mol:
                    props['num_atoms'] = int(mol.GetNumAtoms())
                    props['num_bonds'] = int(mol.GetNumBonds())
            except Exception:
                pass
        for k in measure_keys:
            props.setdefault(k, 0.0)
        if embedder is not None:
            try:
                emb = embedder.embed([smiles])[0]
                for i, v in enumerate(emb):
                    props[f'emb_{i}'] = float(v)
            except Exception:
                pass
        return props

    opt = op.MOCMAMaeOptimizer(
        scheduler=scheduler,
        result_archive=result_archive,
        embedder=embedder,
        mutate_fn=generator.mutate_as_smiles,
        generate_fn=_make_generate_fn(generator),
        evaluate_fn=eval_fn,
        objective_keys=objective_keys,
        measure_keys=measure_keys,
        optimize_objectives=optimize_objectives,
        reference_point=reference_point,
        output_dir=args.output_dir,
        random_init_size=min(args.init_size, args.budget // 5),
    )

    try:
        opt.run(
            n_generations=args.budget,
            log_frequency=args.log_frequency,
            save_frequency=args.save_frequency,
        )
    except BudgetExhaustedError:
        pass


def _run_cma_mae(args, oracle: PMOOracle, generator: MoleculeGenerator,
                 evaluate_fn) -> None:
    """Run (single-objective) CMA-MAE against a PMO oracle (ChemBERTa-UMAP space)."""
    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        'cma_mae_optimizer',
        str(_REPO_ROOT / 'algorithims' / 'cma_mae' / 'optimizer.py'),
    )
    op = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(op)

    from scipy.cluster.vq import kmeans2
    from ribs.archives import CVTArchive
    from ribs.emitters import EvolutionStrategyEmitter
    from ribs.schedulers import Scheduler

    # Fit embedder — provides both the search space and CVT behavioral measures
    embedder = _build_embedder(args, generator)
    measure_keys = embedder.get_measure_keys()
    measure_bounds = embedder.get_measure_bounds()
    cvt_seed_data = embedder.get_fitted_embeddings()
    embed_dim = embedder.n_components

    print(f"Computing CVT centroids ({args.n_centroids} cells from embedding data)...")
    centroids, _ = kmeans2(cvt_seed_data, args.n_centroids, seed=args.seed,
                           minit='points', iter=100)

    cma_archive = CVTArchive(
        solution_dim=embed_dim,
        centroids=centroids,
        ranges=measure_bounds,
        learning_rate=args.learning_rate,
        threshold_min=0.0,
        seed=args.seed,
    )
    result_archive = CVTArchive(
        solution_dim=embed_dim,
        centroids=centroids,
        ranges=measure_bounds,
        learning_rate=1.0,
        seed=args.seed,
    )

    emitters = [
        EvolutionStrategyEmitter(
            archive=cma_archive,
            x0=np.zeros(embed_dim),
            sigma0=args.sigma0,
            batch_size=args.cma_batch_size,
            seed=args.seed + i,
            ranker='imp',
        )
        for i in range(args.n_emitters)
    ]
    scheduler = Scheduler(cma_archive, emitters)

    def eval_fn(smiles):
        props = evaluate_fn(smiles)
        for k in measure_keys:
            props.setdefault(k, 0.0)
        if embedder is not None:
            try:
                emb = embedder.embed([smiles])[0]
                for i, v in enumerate(emb):
                    props[f'emb_{i}'] = float(v)
            except Exception:
                pass
        return props

    opt = op.CMAMaeOptimizer(
        scheduler=scheduler,
        result_archive=result_archive,
        embedder=embedder,
        mutate_fn=generator.mutate_as_smiles,
        generate_fn=_make_generate_fn(generator),
        evaluate_fn=eval_fn,
        measure_keys=measure_keys,
        objective_key=args.oracle,
        output_dir=args.output_dir,
        random_init_size=min(args.init_size, args.budget // 5),
    )

    try:
        opt.run(
            n_generations=args.budget,
            log_frequency=args.log_frequency,
            save_frequency=args.save_frequency,
        )
    except BudgetExhaustedError:
        pass


def _run_mu_lambda(args, oracle: PMOOracle, generator: MoleculeGenerator,
                   evaluate_fn) -> None:
    """Run (μ+λ) evolution strategy against a PMO oracle.

    Inline implementation so BudgetExhaustedError propagates cleanly without
    being swallowed by the existing MuLambdaOptimizer's broad except clauses.
    """
    mu = getattr(args, 'mu', 30)
    lam = getattr(args, 'lam', 30)

    generate_fn = _make_generate_fn(generator)

    # ── initialise population ─────────────────────────────────────────────
    population: list = []   # list of (score, smiles)
    try:
        init_n = min(mu, args.init_size)
        pool = generator.generate_initial_population(init_n)
        for s in pool:
            smiles = generator.decode_to_smiles(s)
            if smiles is None:
                continue
            props = evaluate_fn(smiles)
            score = float(props.get(args.oracle, 0.0) or 0.0)
            population.append((score, smiles))
        population.sort(reverse=True)
        population = population[:mu]
        if not population:
            return
    except BudgetExhaustedError:
        return

    # ── generational loop ─────────────────────────────────────────────────
    try:
        while True:
            offspring: list = []
            for _ in range(lam):
                # Tournament selection (k=3) from top-μ
                candidates = random.choices(population, k=min(3, len(population)))
                _, parent_smiles = max(candidates, key=lambda x: x[0])

                child = generator.mutate_as_smiles(parent_smiles)
                if child is None:
                    child = generate_fn()
                if child is None:
                    continue

                props = evaluate_fn(child)
                score = float(props.get(args.oracle, 0.0) or 0.0)
                offspring.append((score, child))

            # (μ+λ): combine parents + offspring, keep best μ
            combined = population + offspring
            combined.sort(reverse=True)
            population = combined[:mu]
    except BudgetExhaustedError:
        pass


def _run_smiles_lstm_hc(args, oracle: PMOOracle, generator: MoleculeGenerator,
                        evaluate_fn) -> None:
    """Run SMILES-LSTM-HC (Hill Climbing, Brown 2019) via mol_opt bridge.

    Pretrained SMILES LSTM is fine-tuned each round on the top-k molecules seen
    so far, then used to sample new candidates.  One of the strongest PMO baselines
    in Gao et al. 2022.
    """
    import yaml
    sys.path.insert(0, str(_REPO_ROOT / 'algorithims' / 'pmo'))
    _MOL_OPT = '/home/dominic/mol_opt'
    sys.path.insert(0, _MOL_OPT)
    sys.path.insert(0, f'{_MOL_OPT}/main/smiles_lstm_hc')  # rnn_generator, rnn_utils, etc.

    from molopt_bridge import PMOOracleBridge, load_zinc_smiles, make_bridged_optimizer
    from main.smiles_lstm_hc.run import SMILES_LSTM_HC_Optimizer

    bridge = PMOOracleBridge(oracle, args.oracle, args.budget,
                             freq_log=args.log_frequency)
    all_smiles = load_zinc_smiles()

    config_path = f'{_MOL_OPT}/main/smiles_lstm_hc/hparams_default.yaml'
    with open(config_path) as fh:
        config = yaml.safe_load(fh)

    opt = make_bridged_optimizer(SMILES_LSTM_HC_Optimizer, bridge, all_smiles, args)
    opt._optimize(None, config)


def _run_graph_ga(args, oracle: PMOOracle, generator: MoleculeGenerator,
                  evaluate_fn) -> None:
    """Run Graph-GA (Jensen 2019) against a PMO oracle via mol_opt bridge."""
    import yaml
    sys.path.insert(0, str(_REPO_ROOT / 'algorithims' / 'pmo'))
    _MOL_OPT = '/home/dominic/mol_opt'
    sys.path.insert(0, _MOL_OPT)

    from molopt_bridge import PMOOracleBridge, load_zinc_smiles, make_bridged_optimizer
    from main.graph_ga.run import GB_GA_Optimizer

    bridge = PMOOracleBridge(oracle, args.oracle, args.budget,
                             freq_log=args.log_frequency)
    all_smiles = load_zinc_smiles()

    config_path = f'{_MOL_OPT}/main/graph_ga/hparams_default.yaml'
    with open(config_path) as fh:
        config = yaml.safe_load(fh)

    opt = make_bridged_optimizer(GB_GA_Optimizer, bridge, all_smiles, args)
    opt._optimize(None, config)


def _run_smiles_ga(args, oracle: PMOOracle, generator: MoleculeGenerator,
                   evaluate_fn) -> None:
    """Run SMILES-GA (Yoshikawa 2018) against a PMO oracle via mol_opt bridge."""
    import yaml
    sys.path.insert(0, str(_REPO_ROOT / 'algorithims' / 'pmo'))
    _MOL_OPT = '/home/dominic/mol_opt'
    sys.path.insert(0, _MOL_OPT)
    sys.path.insert(0, f'{_MOL_OPT}/main/smiles_ga')  # cfg_util, smiles_grammar

    from molopt_bridge import PMOOracleBridge, load_zinc_smiles, make_bridged_optimizer
    from main.smiles_ga.run import SMILES_GA_Optimizer

    bridge = PMOOracleBridge(oracle, args.oracle, args.budget,
                             freq_log=args.log_frequency)
    all_smiles = load_zinc_smiles()
    # smiles_ga filters out SMILES with '%' (ring-closure tokens beyond 9)
    all_smiles = [s for s in all_smiles if '%' not in s]

    config_path = f'{_MOL_OPT}/main/smiles_ga/hparams_default.yaml'
    with open(config_path) as fh:
        config = yaml.safe_load(fh)

    opt = make_bridged_optimizer(SMILES_GA_Optimizer, bridge, all_smiles, args)
    opt._optimize(None, config)


def _run_reinvent(args, oracle: PMOOracle, generator: MoleculeGenerator,
                  evaluate_fn) -> None:
    """Run REINVENT (Olivecrona 2017 / PMO version) against a PMO oracle via mol_opt bridge."""
    import yaml
    sys.path.insert(0, str(_REPO_ROOT / 'algorithims' / 'pmo'))
    _MOL_OPT = '/home/dominic/mol_opt'
    sys.path.insert(0, _MOL_OPT)
    sys.path.insert(0, f'{_MOL_OPT}/main/reinvent')  # model, data_structs, utils

    from molopt_bridge import PMOOracleBridge, load_zinc_smiles, make_bridged_optimizer
    from main.reinvent.run import REINVENT_Optimizer

    bridge = PMOOracleBridge(oracle, args.oracle, args.budget,
                             freq_log=args.log_frequency)
    all_smiles = load_zinc_smiles()

    config_path = f'{_MOL_OPT}/main/reinvent/hparams_default.yaml'
    with open(config_path) as fh:
        config = yaml.safe_load(fh)

    opt = make_bridged_optimizer(REINVENT_Optimizer, bridge, all_smiles, args)
    opt._optimize(None, config)


def _run_simulated_annealing(args, oracle: PMOOracle, generator: MoleculeGenerator,
                              evaluate_fn) -> None:
    """Run simulated annealing against a PMO oracle.

    Single-trajectory Metropolis SA with geometric cooling.
    Restarts from a new random molecule whenever stuck for too long.
    """
    import math

    T_init = getattr(args, 'sa_T_init', 1.0)
    alpha = getattr(args, 'sa_cooling', 0.999)
    T_min = getattr(args, 'sa_T_min', 1e-4)
    patience = getattr(args, 'sa_patience', 200)  # restarts after this many rejections

    generate_fn = _make_generate_fn(generator)

    def _init_molecule():
        """Evaluate a fresh random molecule. Returns (score, smiles)."""
        for _ in range(20):
            smiles = generate_fn()
            if smiles:
                props = evaluate_fn(smiles)
                score = float(props.get(args.oracle, 0.0) or 0.0)
                return score, smiles
        return 0.0, None

    # ── initialise ────────────────────────────────────────────────────────
    try:
        score, smiles = _init_molecule()
        if smiles is None:
            return
    except BudgetExhaustedError:
        return

    T = T_init
    no_improve = 0

    # ── annealing loop ────────────────────────────────────────────────────
    try:
        while True:
            new_smiles = generator.mutate_as_smiles(smiles)
            if new_smiles is None:
                T = max(T_min, T * alpha)
                continue

            props = evaluate_fn(new_smiles)
            new_score = float(props.get(args.oracle, 0.0) or 0.0)

            delta = new_score - score
            if delta > 0 or (T > T_min and random.random() < math.exp(delta / T)):
                smiles = new_smiles
                score = new_score
                no_improve = 0 if delta > 0 else no_improve + 1
            else:
                no_improve += 1

            T = max(T_min, T * alpha)

            # Restart from new random molecule if stuck
            if no_improve >= patience:
                score, smiles = _init_molecule()
                if smiles is None:
                    break
                T = T_init * 0.5   # restart at half initial temperature
                no_improve = 0
    except BudgetExhaustedError:
        pass


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='PMO benchmark: run an optimizer against a PMO oracle.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Oracle / benchmark settings
    parser.add_argument('--oracle', type=str, required=True,
                        help='PMO oracle name (e.g. qed, penalized_logp, sa, jnk3)')
    parser.add_argument('--budget', type=int, default=10_000,
                        help='Evaluation budget (default: 10000)')
    parser.add_argument('--checkpoint-interval', type=int, default=100,
                        help='AUC checkpoint interval in evaluations (default: 100)')

    # Algorithm
    parser.add_argument('--algorithm', type=str, default='map_elites',
                        choices=['map_elites', 'mome', 'cma_mae', 'mo_cma_mae',
                                 'mu_lambda', 'simulated_annealing',
                                 'graph_ga', 'smiles_ga', 'reinvent', 'smiles_lstm_hc'],
                        help='Optimizer to benchmark (default: map_elites)')

    # Molecule generation
    parser.add_argument('--atom-set', type=str, default='drug',
                        choices=['nlo', 'drug'],
                        help='Atom set for mutation (default: drug)')
    parser.add_argument('--init-size', type=int, default=200,
                        help='Initial archive seeding size (default: 200)')

    # Archive
    parser.add_argument('--archive-type', type=str, default='grid',
                        choices=['grid', 'cvt'],
                        help='Archive type for MOME/MAP-Elites (default: grid)')
    parser.add_argument('--n-centroids', type=int, default=100,
                        help='CVT cells (default: 100)')

    # Embedding (ChemBERTa-UMAP, for cma_mae / mo_cma_mae)
    parser.add_argument('--embedding-model', type=str,
                        default='DeepChem/ChemBERTa-77M-MTR',
                        help='HuggingFace model for ChemBERTa embeddings (default: ChemBERTa-77M-MTR)')
    parser.add_argument('--embedding-dims', type=int, default=10,
                        help='UMAP output dimensionality (default: 10)')
    parser.add_argument('--embedding-device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda', 'mps'],
                        help='Device for ChemBERTa inference (default: auto)')
    parser.add_argument('--embedding-sample-size', type=int, default=5000,
                        help='Molecules used to fit UMAP (default: 5000)')

    # mu+lambda ES parameters
    parser.add_argument('--mu', type=int, default=30,
                        help='(μ+λ) parent population size μ (default: 30)')
    parser.add_argument('--lam', type=int, default=30,
                        help='(μ+λ) offspring count λ per generation (default: 30)')

    # Simulated annealing parameters
    parser.add_argument('--sa-T-init', type=float, default=1.0,
                        help='SA initial temperature (default: 1.0)')
    parser.add_argument('--sa-cooling', type=float, default=0.999,
                        help='SA geometric cooling factor α (default: 0.999)')
    parser.add_argument('--sa-T-min', type=float, default=1e-4,
                        help='SA minimum temperature floor (default: 1e-4)')
    parser.add_argument('--sa-patience', type=int, default=200,
                        help='SA steps without improvement before restart (default: 200)')

    # GA / REINVENT patience (early stopping for mol_opt baselines)
    parser.add_argument('--ga-patience', type=int, default=5,
                        help='Early-stopping patience for graph_ga / smiles_ga / reinvent '
                             '(generations without top-100 improvement, default: 5)')

    # CMA-ES (for mo_cma_mae / cma_mae)
    parser.add_argument('--cma-batch-size', type=int, default=36,
                        help='CMA-ES batch size per emitter (default: 36)')
    parser.add_argument('--n-emitters', type=int, default=5,
                        help='Number of CMA-ES emitters (default: 5)')
    parser.add_argument('--sigma0', type=float, default=0.5,
                        help='CMA-ES initial step size (default: 0.5)')
    parser.add_argument('--learning-rate', type=float, default=0.01,
                        help='CMA-MAE learning rate (default: 0.01)')

    # Run settings
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: pmo_results/<oracle>_<alg>_<seed>)')
    parser.add_argument('--log_frequency', type=int, default=100,
                        help='Logging frequency in generations (default: 100)')
    parser.add_argument('--save_frequency', type=int, default=500,
                        help='Archive save frequency in generations (default: 500)')

    args = parser.parse_args()

    # Default output dir
    if args.output_dir is None:
        args.output_dir = f'pmo_results/{args.oracle}_{args.algorithm}_seed{args.seed}'

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    _seed_everything(args.seed)

    print(f"\n{'='*60}")
    print(f"PMO Benchmark")
    print(f"  Oracle:    {args.oracle}")
    print(f"  Algorithm: {args.algorithm}")
    print(f"  Budget:    {args.budget:,}")
    print(f"  Seed:      {args.seed}")
    print(f"  Output:    {args.output_dir}")
    print(f"{'='*60}\n")

    # Create oracle
    oracle = create_oracle(
        args.oracle,
        budget=args.budget,
        checkpoint_interval=args.checkpoint_interval,
    )

    # Molecule generator
    generator = MoleculeGenerator(seed=args.seed, atom_set=args.atom_set)

    evaluate_fn = _make_evaluate_fn(oracle)

    # Run chosen algorithm
    if args.algorithm == 'map_elites':
        _run_map_elites(args, oracle, generator, evaluate_fn)
    elif args.algorithm == 'mome':
        _run_mome(args, oracle, generator, evaluate_fn)
    elif args.algorithm == 'cma_mae':
        _run_cma_mae(args, oracle, generator, evaluate_fn)
    elif args.algorithm == 'mo_cma_mae':
        _run_mo_cma_mae(args, oracle, generator, evaluate_fn)
    elif args.algorithm == 'mu_lambda':
        _run_mu_lambda(args, oracle, generator, evaluate_fn)
    elif args.algorithm == 'simulated_annealing':
        _run_simulated_annealing(args, oracle, generator, evaluate_fn)
    elif args.algorithm == 'graph_ga':
        _run_graph_ga(args, oracle, generator, evaluate_fn)
    elif args.algorithm == 'smiles_ga':
        _run_smiles_ga(args, oracle, generator, evaluate_fn)
    elif args.algorithm == 'reinvent':
        _run_reinvent(args, oracle, generator, evaluate_fn)
    elif args.algorithm == 'smiles_lstm_hc':
        _run_smiles_lstm_hc(args, oracle, generator, evaluate_fn)

    # Save PMO results
    auc = oracle.get_auc_scores()
    oracle.save_results(args.output_dir)

    print(f"\n{'='*60}")
    print(f"PMO Results — {args.oracle} / {args.algorithm} / seed={args.seed}")
    print(f"  Top-1  (final): {auc['final_top1']:.4f}   AUC: {auc['top1_auc']:.4f}")
    print(f"  Top-10 (final): {auc['final_top10']:.4f}   AUC: {auc['top10_auc']:.4f}")
    print(f"  Top-100(final): {auc['final_top100']:.4f}   AUC: {auc['top100_auc']:.4f}")
    print(f"  Valid evaluations: {auc['n_valid']} / {auc['n_evaluations']}")
    print(f"{'='*60}\n")

    # Save summary
    summary = {
        'oracle': args.oracle,
        'algorithm': args.algorithm,
        'seed': args.seed,
        'budget': args.budget,
        **auc,
    }
    with open(Path(args.output_dir) / 'pmo_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Results saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
