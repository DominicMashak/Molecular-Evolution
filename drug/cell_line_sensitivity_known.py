#!/usr/bin/env python3
"""
cell_line_sensitivity_known.py

For each cell line, counts how many times it has:
  - The LOWEST known IC50 (most potent) across all drugs tested on it
  - The HIGHEST known IC50 (least potent) across all drugs tested on it

Uses actual GDSC training data, no model inference.

Usage:
    python cell_line_sensitivity_known.py
    python cell_line_sensitivity_known.py --output known_sensitivity.csv
"""

import argparse
import csv
import numpy as np
from collections import defaultdict


#DRUG_IC50_PATH = "/Users/rohanbasuroy/Documents/GitHub/GPDRP/data/drug_cl_ic.csv"
DRUG_IC50_PATH = "/Users/rohanbasuroy/Documents/GitHub/GPDRP_GDSC2/data/drug_cl_ic.csv"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="cell_line_sensitivity_known.csv")
    args = parser.parse_args()

    # load all data into drug -> {cell_line: ic50} mapping
    print("Loading known IC50 values...")
    drug_data = defaultdict(dict)

    with open(DRUG_IC50_PATH, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            drug      = row['Drug name'].strip()
            cell_line = row['Cell line name'].strip()
            try:
                ic50 = float(row['IC50'])
                drug_data[drug][cell_line] = ic50
            except ValueError:
                pass

    print(f"  {len(drug_data)} drugs loaded")

    # counters
    most_potent_count  = defaultdict(int)
    least_potent_count = defaultdict(int)

    for drug_name, cell_ic50s in drug_data.items():
        if not cell_ic50s:
            continue

        best_cell  = min(cell_ic50s, key=cell_ic50s.get)  # lowest IC50 = most potent
        worst_cell = max(cell_ic50s, key=cell_ic50s.get)  # highest IC50 = least potent

        most_potent_count[best_cell]   += 1
        least_potent_count[worst_cell] += 1

    # combine — ignore zeros
    all_cell_lines = set(most_potent_count.keys()) | set(least_potent_count.keys())

    results = []
    for cell in all_cell_lines:
        most  = most_potent_count[cell]
        least = least_potent_count[cell]
        if most == 0 and least == 0:
            continue
        results.append({
            'cell_line':          cell,
            'most_potent_count':  most,
            'least_potent_count': least,
            'net_sensitivity':    most - least
        })

    # sort by most potent count
    results_by_most  = sorted(results, key=lambda x: x['most_potent_count'],  reverse=True)
    results_by_least = sorted(results, key=lambda x: x['least_potent_count'], reverse=True)

    # assign ranks
    for i, r in enumerate(results_by_most):
        r['rank_most_potent'] = i + 1
    for i, r in enumerate(results_by_least):
        r['rank_least_potent'] = i + 1

    # print summary
    print(f"\n{'='*65}")
    print("KNOWN CELL LINE SENSITIVITY (GDSC training data)")
    print(f"{'='*65}")

    print(f"\nTop 15 most frequently MOST POTENT cell lines:")
    print(f"{'Cell line':<20} {'Most potent':>12} {'Least potent':>13} {'Net':>6}")
    print(f"{'-'*53}")
    for r in results_by_most[:15]:
        print(f"{r['cell_line']:<20} {r['most_potent_count']:>12} "
              f"{r['least_potent_count']:>13} {r['net_sensitivity']:>6}")

    print(f"\nTop 15 most frequently LEAST POTENT cell lines:")
    print(f"{'Cell line':<20} {'Most potent':>12} {'Least potent':>13} {'Net':>6}")
    print(f"{'-'*53}")
    for r in results_by_least[:15]:
        print(f"{r['cell_line']:<20} {r['most_potent_count']:>12} "
              f"{r['least_potent_count']:>13} {r['net_sensitivity']:>6}")

    # save
    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'cell_line', 'most_potent_count', 'least_potent_count',
            'net_sensitivity', 'rank_most_potent', 'rank_least_potent'
        ])
        writer.writeheader()
        writer.writerows(results_by_most)

    print(f"\nFull results saved to {args.output}")
    print(f"Total cell lines with non-zero counts: {len(results)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)