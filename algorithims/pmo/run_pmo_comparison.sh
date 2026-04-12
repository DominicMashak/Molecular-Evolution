#!/usr/bin/env bash
# =============================================================================
# PMO Benchmark Comparison Suite
# =============================================================================
#
# Runs all implemented algorithms on a standard oracle set with multiple seeds,
# producing results directly comparable to published PMO leaderboard numbers
# (Gao et al., NeurIPS 2022 — https://arxiv.org/abs/2206.12411).
#
# Single-objective algorithms (PMO is single-objective; mome/mo_cma_mae excluded):
#   map_elites          MAP-Elites (QD, grid archive, structural measures)
#   cma_mae             CMA-MAE (QD + CMA-ES, CVT archive, ChemBERTa UMAP space)
#   mu_lambda           (μ+λ) Evolution Strategy (no archive, simple population ES)
#   simulated_annealing Simulated Annealing (single trajectory + restarts)
#   graph_ga            Graph-GA      (Jensen 2019) via mol_opt bridge
#   smiles_ga           SMILES-GA     (Yoshikawa 2018) via mol_opt bridge
#   reinvent            REINVENT      (Olivecrona 2017 / PMO version) via mol_opt bridge
#   smiles_lstm_hc      SMILES-LSTM-HC (Brown 2019, top PMO baseline) via mol_opt bridge
#
# mol_opt repo must be present at /home/dominic/mol_opt (already cloned).
# mol-evo conda env is used for all runs (tdc is installed there).
#
# Oracle set:
#   Local (no TDC needed): qed, penalized_logp, sa
#   TDC-backed (pip install PyTDC): jnk3, gsk3b, drd2, osimertinib_mpo
#
# Usage:
#   bash run_pmo_comparison.sh                  # full suite (all oracles, all algorithms)
#   bash run_pmo_comparison.sh qed              # single oracle, all algorithms
#   bash run_pmo_comparison.sh qed cma_mae      # single oracle + algorithm
#   bash run_pmo_comparison.sh qed cma_mae 10000 42   # oracle, alg, budget, seed
#
# =============================================================================

set -euo pipefail

PYTHON=/home/dominic/miniconda3/envs/mol-evo/bin/python
SCRIPT="$(dirname "$0")/run_pmo_benchmark.py"

# ── Configuration ─────────────────────────────────────────────────────────────
# Oracles: start with local-only set; append TDC oracles if PyTDC is installed.
#   Local:  qed  penalized_logp  sa
#   TDC:    jnk3  gsk3b  drd2  osimertinib_mpo
LOCAL_ORACLES="qed penalized_logp sa"
TDC_ORACLES="jnk3 gsk3b drd2 osimertinib_mpo"

# Default: local only; set USE_TDC=1 to include TDC oracles
USE_TDC=${USE_TDC:-0}

if [ "$USE_TDC" = "1" ]; then
    DEFAULT_ORACLES="$LOCAL_ORACLES $TDC_ORACLES"
else
    DEFAULT_ORACLES="$LOCAL_ORACLES"
fi

# Single-objective algorithms for PMO comparison
# (mome and mo_cma_mae are multi-objective and not comparable on PMO metrics)
ALL_ALGORITHMS="map_elites cma_mae mu_lambda simulated_annealing graph_ga smiles_ga reinvent smiles_lstm_hc"

ORACLES=${ORACLES:-"$DEFAULT_ORACLES"}
ALGORITHMS=${ALGORITHMS:-"$ALL_ALGORITHMS"}
SEEDS=${SEEDS:-"1 42 99"}
BUDGET=${BUDGET:-10000}

# Override with positional args
[ -n "${1:-}" ] && ORACLES="$1"
[ -n "${2:-}" ] && ALGORITHMS="$2"
[ -n "${3:-}" ] && BUDGET="$3"
[ -n "${4:-}" ] && SEEDS="$4"

# ── Shared hyperparameters ────────────────────────────────────────────────────
ATOM_SET="drug"
INIT_SIZE=200
N_CENTROIDS=100
LOG_FREQ=100
SAVE_FREQ=2000

# CMA-MAE / MO-CMA-MAE
EMBED_DIMS=10
EMBED_SAMPLE=5000
CMA_BATCH=36
N_EMITTERS=5
SIGMA0=0.5
LR=0.01          # threshold annealing learning rate

# (μ+λ) ES
MU=30
LAM=30

# Simulated Annealing
SA_T_INIT=1.0
SA_COOLING=0.999
SA_T_MIN=0.0001
SA_PATIENCE=200

# Graph-GA / SMILES-GA / REINVENT (mol_opt baselines)
GA_PATIENCE=5

# ── Helper: run one (oracle, algorithm, seed) triple ─────────────────────────
run_one() {
    local ORACLE="$1"
    local ALG="$2"
    local SEED="$3"
    local OUTDIR="pmo_results/${ORACLE}_${ALG}_seed${SEED}"

    # Skip if already completed
    if [ -f "${OUTDIR}/pmo_summary.json" ]; then
        echo "  [SKIP] ${OUTDIR} already exists."
        return 0
    fi

    echo ""
    echo "=========================================="
    echo " Oracle: $ORACLE | Algorithm: $ALG | Seed: $SEED"
    echo "=========================================="

    # Base args shared by all algorithms
    BASE_ARGS=(
        --oracle "$ORACLE"
        --algorithm "$ALG"
        --budget "$BUDGET"
        --seed "$SEED"
        --output_dir "$OUTDIR"
        --atom-set "$ATOM_SET"
        --init-size "$INIT_SIZE"
        --n-centroids "$N_CENTROIDS"
        --log_frequency "$LOG_FREQ"
        --save_frequency "$SAVE_FREQ"
    )

    # Algorithm-specific args
    case "$ALG" in
    cma_mae|mo_cma_mae)
        EXTRA_ARGS=(
            --embedding-dims "$EMBED_DIMS"
            --embedding-sample-size "$EMBED_SAMPLE"
            --cma-batch-size "$CMA_BATCH"
            --n-emitters "$N_EMITTERS"
            --sigma0 "$SIGMA0"
            --learning-rate "$LR"
        )
        ;;
    mu_lambda)
        EXTRA_ARGS=(
            --mu "$MU"
            --lam "$LAM"
        )
        ;;
    simulated_annealing)
        EXTRA_ARGS=(
            --sa-T-init "$SA_T_INIT"
            --sa-cooling "$SA_COOLING"
            --sa-T-min "$SA_T_MIN"
            --sa-patience "$SA_PATIENCE"
        )
        ;;
    graph_ga|smiles_ga|reinvent)
        EXTRA_ARGS=(
            --ga-patience "$GA_PATIENCE"
        )
        ;;
    *)
        EXTRA_ARGS=()
        ;;
    esac

    $PYTHON "$SCRIPT" "${BASE_ARGS[@]}" "${EXTRA_ARGS[@]}"
    echo "Done: $OUTDIR"
}

# ── Main loop ─────────────────────────────────────────────────────────────────
TOTAL=0
for ORACLE in $ORACLES; do
    for ALG in $ALGORITHMS; do
        for SEED in $SEEDS; do
            TOTAL=$((TOTAL + 1))
        done
    done
done

echo ""
echo "PMO Comparison Suite"
echo "  Oracles:    $ORACLES"
echo "  Algorithms: $ALGORITHMS"
echo "  Seeds:      $SEEDS"
echo "  Budget:     $BUDGET calls each"
echo "  Total runs: $TOTAL"
echo ""

COMPLETED=0
for ORACLE in $ORACLES; do
    for ALG in $ALGORITHMS; do
        for SEED in $SEEDS; do
            COMPLETED=$((COMPLETED + 1))
            echo "[${COMPLETED}/${TOTAL}] Starting $ORACLE / $ALG / seed=$SEED"
            run_one "$ORACLE" "$ALG" "$SEED"
        done
    done
done

echo ""
echo "All runs complete."
echo ""

# ── Aggregate results table ───────────────────────────────────────────────────
echo "Aggregating results..."
$PYTHON - <<'PYEOF'
import json, glob, collections
import numpy as np
from pathlib import Path

results = collections.defaultdict(list)
for f in sorted(glob.glob('pmo_results/*/pmo_summary.json')):
    try:
        with open(f) as fp:
            d = json.load(fp)
        key = (d['oracle'], d['algorithm'])
        results[key].append(d)
    except Exception:
        pass

if not results:
    print("No results found in pmo_results/")
    exit(0)

# Metrics to report
METRICS = [
    ('top1_auc',    'Top1-AUC'),
    ('top10_auc',   'Top10-AUC'),
    ('top100_auc',  'Top100-AUC'),
    ('final_top1',  'Top1-Final'),
    ('final_top10', 'Top10-Final'),
]

# Gather all oracles and algorithms present
oracles = sorted({k[0] for k in results})
algorithms = sorted({k[1] for k in results})

print(f"\n{'='*90}")
print(f"{'PMO BENCHMARK RESULTS':^90}")
print(f"{'='*90}")

for oracle in oracles:
    print(f"\n  Oracle: {oracle}")
    hdr = f"  {'Algorithm':<22}"
    for _, label in METRICS:
        hdr += f"  {label:>12}"
    hdr += "  Seeds"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for alg in algorithms:
        key = (oracle, alg)
        if key not in results:
            continue
        runs = results[key]
        row = f"  {alg:<22}"
        for metric, _ in METRICS:
            vals = [r[metric] for r in runs if metric in r]
            if vals:
                row += f"  {np.mean(vals):8.4f}±{np.std(vals):.3f}"
            else:
                row += f"  {'—':>12}"
        row += f"  {len(runs):5d}"
        print(row)

print(f"\n{'='*90}")
print("")
PYEOF

echo "Results saved in pmo_results/"
echo "Run 'python run_pmo_benchmark.py --help' for single-run options."
