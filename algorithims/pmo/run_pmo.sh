#!/usr/bin/env bash
# PMO benchmark: run all algorithms on standard oracles with multiple seeds.
# Edit ORACLES, ALGORITHMS, SEEDS, and BUDGET below to customise.
#
# Usage:
#   bash run_pmo.sh                          # full suite
#   bash run_pmo.sh qed mome 3000 42         # oracle alg budget seed

set -euo pipefail

PYTHON=/home/dominic/miniconda3/envs/mol-evo/bin/python
SCRIPT="$(dirname "$0")/run_pmo_benchmark.py"

# ── configuration ────────────────────────────────────────────────────────────
ORACLES=${ORACLES:-"qed penalized_logp sa"}
ALGORITHMS=${ALGORITHMS:-"mome map_elites"}
SEEDS=${SEEDS:-"1 42 99"}
BUDGET=${BUDGET:-10000}

# Override with positional args if provided
[ -n "${1:-}" ] && ORACLES="$1"
[ -n "${2:-}" ] && ALGORITHMS="$2"
[ -n "${3:-}" ] && BUDGET="$3"
[ -n "${4:-}" ] && SEEDS="$4"

# ── run ───────────────────────────────────────────────────────────────────────
for ORACLE in $ORACLES; do
  for ALG in $ALGORITHMS; do
    for SEED in $SEEDS; do
      OUTDIR="pmo_results/${ORACLE}_${ALG}_seed${SEED}"
      echo ""
      echo "=========================================="
      echo " Oracle: $ORACLE | Algorithm: $ALG | Seed: $SEED"
      echo "=========================================="
      $PYTHON "$SCRIPT" \
        --oracle "$ORACLE" \
        --algorithm "$ALG" \
        --budget "$BUDGET" \
        --seed "$SEED" \
        --output_dir "$OUTDIR" \
        --atom-set drug \
        --archive-type grid \
        --n-centroids 100 \
        --init-size 200 \
        --log_frequency 100 \
        --save_frequency 1000
      echo "Done: $OUTDIR"
    done
  done
done

echo ""
echo "All PMO runs complete."
echo "Results in pmo_results/"

# ── aggregate results ─────────────────────────────────────────────────────────
echo ""
echo "Aggregating results..."
$PYTHON - <<'EOF'
import json, glob, collections
from pathlib import Path

results = collections.defaultdict(list)
for f in sorted(glob.glob('pmo_results/*/pmo_summary.json')):
    with open(f) as fp:
        d = json.load(fp)
    key = f"{d['oracle']}|{d['algorithm']}"
    results[key].append(d)

print(f"\n{'Oracle':<20} {'Algorithm':<12} {'Seeds'} {'Top10-AUC':>12} {'Top10-Final':>12}")
print("-" * 65)
import numpy as np
for key, runs in sorted(results.items()):
    oracle, alg = key.split('|')
    aucs = [r['top10_auc'] for r in runs]
    finals = [r['final_top10'] for r in runs]
    print(f"{oracle:<20} {alg:<12} {len(runs):5d}"
          f" {np.mean(aucs):12.4f} ± {np.std(aucs):.4f}"
          f" {np.mean(finals):12.4f} ± {np.std(finals):.4f}")
EOF
