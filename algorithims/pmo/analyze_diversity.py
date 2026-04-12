"""
analyze_diversity.py — Compute chemical diversity metrics across PMO benchmark runs.

Reads molecule databases from completed benchmark runs and reports:

  Metric               Description
  ──────────────────── ─────────────────────────────────────────────────────────
  IntDiv1              Mean pairwise Tanimoto distance in top-100 (1 = max diverse)
  IntDiv2              Sqrt variant (down-weights outliers; standard in MOSES)
  n_scaffolds_100      Unique Bemis-Murcko scaffolds in top-100
  scaffold_ratio_100   n_scaffolds / 100  (1.0 = every mol has a unique scaffold)
  n_scaffolds_1000     Unique Bemis-Murcko scaffolds in top-1000
  top1 / top10 / top100  Primary oracle scores (from pmo_summary.json)

Molecule sources (in priority order):
  1. <oracle>_top_molecules.json  — saved by PMOOracle.save_results() (all algorithms)
  2. all_molecules_database.json  — saved by MOME / MAP-Elites optimizers

Usage
─────
  # Analyse all runs in pmo_results/
  python analyze_diversity.py --results-dir pmo_results/

  # Analyse specific run
  python analyze_diversity.py --results-dir pmo_results/qed_mome_42

  # Change top-N cutoff
  python analyze_diversity.py --results-dir pmo_results/ --top-n 100
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent / 'molev_utils'))


# ──────────────────────────────────────────────────────────────────────────────
# RDKit helpers
# ──────────────────────────────────────────────────────────────────────────────

def _morgan_fps(smiles_list: List[str], radius: int = 2, nbits: int = 2048):
    """Return list of Morgan fingerprint bit-vectors (None for invalid SMILES)."""
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fps.append(None)
        else:
            fps.append(rdMolDescriptors.GetMorganFingerprintAsBitVect(
                mol, radius, nbits))
    return fps


def _tanimoto_intdiv(fps, variant: int = 1) -> float:
    """
    Internal diversity = mean pairwise Tanimoto distance (1 − similarity).

    variant=1: arithmetic mean
    variant=2: sqrt of mean of squared distances (MOSES IntDiv2)
    """
    from rdkit.DataStructs import TanimotoSimilarity
    valid = [fp for fp in fps if fp is not None]
    n = len(valid)
    if n < 2:
        return 0.0

    distances = []
    for i in range(n):
        for j in range(i + 1, n):
            sim = TanimotoSimilarity(valid[i], valid[j])
            distances.append(1.0 - sim)

    if variant == 2:
        return float(np.sqrt(np.mean(np.array(distances) ** 2)))
    return float(np.mean(distances))


def _bemis_murcko_scaffolds(smiles_list: List[str]) -> List[Optional[str]]:
    """Return canonical Bemis-Murcko scaffold SMILES for each input SMILES."""
    from rdkit import Chem
    from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
    scaffolds = []
    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                scaffolds.append(None)
            else:
                sc = MurckoScaffoldSmiles(mol=mol)
                scaffolds.append(sc)
        except Exception:
            scaffolds.append(None)
    return scaffolds


def compute_diversity(smiles_list: List[str], top_n: int = 100) -> dict:
    """
    Compute diversity metrics on top_n molecules.

    Parameters
    ----------
    smiles_list : ordered list, best first (index 0 = highest score)
    top_n : how many top molecules to analyse

    Returns
    -------
    dict with keys: intdiv1, intdiv2, n_scaffolds, scaffold_ratio,
                    n_valid, top_n_used
    """
    candidates = smiles_list[:top_n]

    # Validate
    from rdkit import Chem
    valid = [s for s in candidates if s and Chem.MolFromSmiles(s) is not None]
    n_valid = len(valid)

    if n_valid == 0:
        return {
            'intdiv1': 0.0, 'intdiv2': 0.0,
            'n_scaffolds': 0, 'scaffold_ratio': 0.0,
            'n_valid': 0, 'top_n_used': top_n,
        }

    fps = _morgan_fps(valid)
    intdiv1 = _tanimoto_intdiv(fps, variant=1)
    intdiv2 = _tanimoto_intdiv(fps, variant=2)

    scaffolds = _bemis_murcko_scaffolds(valid)
    unique_scaffolds = len({s for s in scaffolds if s is not None})

    return {
        'intdiv1': round(intdiv1, 4),
        'intdiv2': round(intdiv2, 4),
        'n_scaffolds': unique_scaffolds,
        'scaffold_ratio': round(unique_scaffolds / max(1, n_valid), 4),
        'n_valid': n_valid,
        'top_n_used': top_n,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def _load_smiles_from_dir(run_dir: Path, oracle: Optional[str] = None
                          ) -> Tuple[List[str], str]:
    """
    Load (score, smiles) pairs from a benchmark result directory.

    Returns (smiles_sorted_best_first, oracle_name).
    """
    # ── 1. PMOOracle top_molecules.json (new format, all algorithms) ──────────
    if oracle:
        top_mol_path = run_dir / f'{oracle}_top_molecules.json'
        if top_mol_path.exists():
            data = json.loads(top_mol_path.read_text())
            sorted_data = sorted(data, key=lambda x: x['score'], reverse=True)
            return [d['smiles'] for d in sorted_data], oracle

    # Auto-detect oracle name from any *_top_molecules.json
    for p in run_dir.glob('*_top_molecules.json'):
        oracle_name = p.stem.replace('_top_molecules', '')
        data = json.loads(p.read_text())
        sorted_data = sorted(data, key=lambda x: x['score'], reverse=True)
        return [d['smiles'] for d in sorted_data], oracle_name

    # ── 2. all_molecules_database.json (MOME / MAP-Elites) ───────────────────
    db_path = run_dir / 'all_molecules_database.json'
    if db_path.exists():
        db = json.loads(db_path.read_text())
        entries = list(db.values()) if isinstance(db, dict) else db

        # Detect oracle key: prefer the one in pmo_summary, else highest-value key
        oracle_name = _detect_oracle(run_dir, entries)

        sorted_entries = sorted(
            entries,
            key=lambda x: x.get(oracle_name, 0.0) or 0.0,
            reverse=True,
        )
        smiles = [e.get('smiles') for e in sorted_entries if e.get('smiles')]
        return smiles, oracle_name

    return [], 'unknown'


def _detect_oracle(run_dir: Path, entries: list) -> str:
    """Infer oracle name from pmo_summary.json or database keys."""
    summary_path = run_dir / 'pmo_summary.json'
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        if 'oracle' in summary:
            return summary['oracle']

    # Fall back: pick the highest-mean numeric key (excluding structural keys)
    exclude = {'num_atoms', 'num_bonds', 'num_atoms_bin', 'num_bonds_bin',
               'generation', 'mol_weight', 'error'}
    if entries:
        keys = [k for k in entries[0] if k not in exclude
                and isinstance(entries[0].get(k), (int, float))]
        if keys:
            means = {k: np.mean([e.get(k, 0) or 0 for e in entries]) for k in keys}
            return max(means, key=means.get)
    return 'objective'


def _load_summary(run_dir: Path) -> dict:
    p = run_dir / 'pmo_summary.json'
    if p.exists():
        return json.loads(p.read_text())
    return {}


# ──────────────────────────────────────────────────────────────────────────────
# Main analysis
# ──────────────────────────────────────────────────────────────────────────────

def analyse_run(run_dir: Path, top_n: int, oracle: Optional[str] = None) -> dict:
    smiles, oracle_name = _load_smiles_from_dir(run_dir, oracle)
    summary = _load_summary(run_dir)

    # Prefer oracle name from pmo_summary.json (more reliable)
    if summary.get('oracle'):
        oracle_name = summary['oracle']

    result = {
        'run': run_dir.name,
        'oracle': oracle_name,
        'algorithm': summary.get('algorithm', '?'),
        'n_evaluations': summary.get('n_evaluations', len(smiles)),
        'top1': summary.get('final_top1'),
        'top10': summary.get('final_top10'),
        'top100': summary.get('final_top100'),
        'top1_auc': summary.get('top1_auc'),
        'n_molecules_available': len(smiles),
    }

    if smiles:
        div = compute_diversity(smiles, top_n=top_n)
        result.update(div)
        # Also compute at top-1000 scaffold count if enough molecules
        if len(smiles) >= 1000:
            sc_1000 = _bemis_murcko_scaffolds(smiles[:1000])
            result['n_scaffolds_1000'] = len({s for s in sc_1000 if s is not None})
        else:
            sc_all = _bemis_murcko_scaffolds(smiles)
            result['n_scaffolds_1000'] = len({s for s in sc_all if s is not None})
    else:
        result['error'] = 'no molecule data found'

    return result


def print_table(results: List[dict], top_n: int) -> None:
    """Print a formatted comparison table to stdout."""
    if not results:
        print("No results to display.")
        return

    # Group by oracle
    oracles = sorted({r['oracle'] for r in results})

    for oracle in oracles:
        rows = [r for r in results if r['oracle'] == oracle]
        if not rows:
            continue

        print(f"\n{'═' * 90}")
        print(f"  Oracle: {oracle.upper()}   (diversity measured on top-{top_n} molecules)")
        print(f"{'═' * 90}")
        print(f"{'Method':<28} {'Top-1':>7} {'Top-10':>7} {'Top-100':>8} "
              f"{'AUC-1':>7} {'IntDiv1':>8} {'IntDiv2':>8} "
              f"{'Scaffolds':>10} {'Sc/N':>6}")
        print(f"{'─' * 90}")

        for r in sorted(rows, key=lambda x: x.get('top1') or 0, reverse=True):
            alg = r.get('algorithm', r['run'])
            top1 = f"{r['top1']:.4f}" if r.get('top1') is not None else '  —  '
            top10 = f"{r['top10']:.4f}" if r.get('top10') is not None else '  —  '
            top100 = f"{r['top100']:.4f}" if r.get('top100') is not None else '  —  '
            auc1 = f"{r['top1_auc']:.4f}" if r.get('top1_auc') is not None else '  —  '
            intd1 = f"{r['intdiv1']:.4f}" if 'intdiv1' in r else '  —  '
            intd2 = f"{r['intdiv2']:.4f}" if 'intdiv2' in r else '  —  '
            nsc = str(r.get('n_scaffolds', '—'))
            scr = f"{r['scaffold_ratio']:.3f}" if 'scaffold_ratio' in r else '  —  '
            print(f"  {alg:<26} {top1:>7} {top10:>7} {top100:>8} "
                  f"{auc1:>7} {intd1:>8} {intd2:>8} {nsc:>10} {scr:>6}")

        print(f"{'─' * 90}")
        print(f"  IntDiv1: mean pairwise Tanimoto distance (1=maximally diverse)")
        print(f"  IntDiv2: sqrt variant — down-weights outlier pairs (MOSES standard)")
        print(f"  Scaffolds: unique Bemis-Murcko scaffolds in top-{top_n}")


def main():
    parser = argparse.ArgumentParser(
        description='Chemical diversity analysis of PMO benchmark runs.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--results-dir', type=str, default='pmo_results',
                        help='Directory containing benchmark result subdirectories '
                             '(default: pmo_results/)')
    parser.add_argument('--top-n', type=int, default=100,
                        help='Number of top molecules to compute diversity over '
                             '(default: 100)')
    parser.add_argument('--oracle', type=str, default=None,
                        help='Filter to a specific oracle (e.g. qed, penalized_logp)')
    parser.add_argument('--output', type=str, default=None,
                        help='Save results to a JSON file (optional)')
    args = parser.parse_args()

    results_path = Path(args.results_dir)

    # Collect run directories
    if (results_path / 'pmo_summary.json').exists():
        # Single run directory passed directly
        run_dirs = [results_path]
    else:
        run_dirs = sorted([
            d for d in results_path.iterdir()
            if d.is_dir() and not d.name.startswith('.')
        ])

    if not run_dirs:
        print(f"No result directories found in {results_path}")
        sys.exit(1)

    print(f"Analysing {len(run_dirs)} run(s) — top-{args.top_n} diversity...")
    results = []
    for run_dir in run_dirs:
        # Skip dirs with no recognisable output
        has_data = (
            (run_dir / 'pmo_summary.json').exists() or
            (run_dir / 'all_molecules_database.json').exists() or
            any(run_dir.glob('*_top_molecules.json'))
        )
        if not has_data:
            continue

        # Filter by oracle if requested
        if args.oracle:
            summary = _load_summary(run_dir)
            if summary.get('oracle', '') != args.oracle:
                continue

        print(f"  {run_dir.name}... ", end='', flush=True)
        try:
            r = analyse_run(run_dir, top_n=args.top_n, oracle=args.oracle)
            results.append(r)
            n = r.get('n_scaffolds', '?')
            d = r.get('intdiv1', '?')
            print(f"IntDiv1={d}  Scaffolds={n}")
        except Exception as e:
            print(f"ERROR: {e}")

    print_table(results, top_n=args.top_n)

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
