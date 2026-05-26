#!/bin/bash
set -euo pipefail

# Critical temperature (Tc) prediction for a SLICES crystal structure.
# Uses pretrained ALIGNN models from JARVIS/NIST (InvDesFlow workflow):
#   - jv_supercon_tc_alignn        → Tc in Kelvin
#   - jv_formation_energy_peratom_alignn → formation energy (eV/atom)
#   - jv_mbj_bandgap               → band gap (eV)
#   - jv_supercon_debye_alignn     → Debye temperature (K)
#
# Screening criteria (Wines et al. 2023 / InvDesFlow):
#   Tc > 5 K  AND  Eform < 0 eV/atom  AND  gap < 0.05 eV
#
# Requires: pip install alignn jarvis-tools
# Models are downloaded from Figshare and cached automatically on first run.
#
# Reference:
#   Han et al., Chinese Physics Letters 42, 047301 (2025)
#   https://github.com/xqh19970407/InvDesFlow

python ~/Molecular-Evolution/quantum_chemistry/crystal_main.py \
    --calculator alignn_tc \
    --slices "Nb Si  0 1 ooo  0 1 +oo  0 1 o+o  0 1 oo+  1 0 ooo" \
    --properties tc tc_formation_energy tc_bandgap tc_debye_temperature is_superconductor_candidate n_sites n_species volume density spacegroup \
    --tc-threshold 5.0
