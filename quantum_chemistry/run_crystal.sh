#!/bin/bash
set -euo pipefail

# Single-material crystal property calculator.
# Uses CHGNet ML surrogate (~1 s) by default; switch to --calculator qe for DFT.
#
# Usage:
#   bash run_crystal.sh                          # run example below
#   bash run_crystal.sh --calculator qe \
#       --pseudo-dir /path/to/pseudos            # full QE DFT
#
# Example SLICES strings:
#   TiO2 rutile-like:  "Ti O O  0 1 ooo  0 2 +oo  1 2 o+o  0 1 oo+"
#   SiO2-like:         "Si O O O  0 1 ooo  0 2 +oo  0 3 o+o  1 2 oo+"
#   NaCl rock-salt:    "Na Cl  0 1 ooo  0 1 +oo  0 1 o+o"

python ~/Molecular-Evolution/quantum_chemistry/crystal_main.py \
    --calculator chgnet \
    --slices "Ti O O  0 1 ooo  0 2 +oo  1 2 o+o  0 1 oo+" \
    --properties formation_energy n_sites n_species volume density spacegroup
