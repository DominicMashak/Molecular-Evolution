#!/usr/bin/env python3
"""
Crystal Property Calculator — CLI for SLICES crystal structures.

Analogous to quantum_chemistry/main.py but operates on periodic crystal
structures encoded as SLICES strings instead of molecular SMILES.

Calculators
-----------
chgnet (default)
    Fast ML surrogate (~1 s).  Predicts formation energy and structural
    descriptors.  Optionally relaxes the structure before evaluation.

qe
    Accurate DFT via Quantum ESPRESSO (~minutes).  Requires pseudopotentials
    on disk (see SYSTEM_REQUIREMENTS.txt) and the qe conda package.
    Computes formation energy, band gap, and effective masses.

Usage
-----
    python crystal_main.py --calculator chgnet \\
        --slices "Ti O O  0 1 ooo  0 2 +oo  1 2 o+o  0 1 oo+" \\
        --properties formation_energy n_sites n_species volume density spacegroup

    python crystal_main.py --calculator qe \\
        --slices "Ti O O  0 1 ooo  0 2 +oo  1 2 o+o  0 1 oo+" \\
        --properties formation_energy bandgap effective_mass_e effective_mass_h \\
        --pseudo-dir /path/to/pseudos
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Path setup — allow imports from molev_utils regardless of cwd
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
_MOLEV_UTILS = _REPO_ROOT / 'molev_utils'
_QC_DIR = Path(__file__).resolve().parent

for _p in (_MOLEV_UTILS, _QC_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# ---------------------------------------------------------------------------
# All properties that can be requested, by calculator
# ---------------------------------------------------------------------------

_CHGNET_PROPS = {
    'formation_energy', 'n_sites', 'n_species', 'volume', 'density',
    'spacegroup', 'spacegroup_number',
}

_QE_PROPS = {
    'formation_energy', 'bandgap', 'bandgap_direct', 'vbm', 'cbm',
    'is_metal', 'effective_mass_e', 'effective_mass_h',
    'n_sites', 'n_species', 'volume', 'density', 'spacegroup',
    'total_energy', 'e_above_hull',
}

_ALIGNN_TC_PROPS = {
    'tc', 'tc_formation_energy', 'tc_bandgap', 'tc_debye_temperature',
    'is_superconductor_candidate', 'n_sites', 'n_species', 'volume',
    'density', 'spacegroup',
}

_ALL_PROPS = _CHGNET_PROPS | _QE_PROPS | _ALIGNN_TC_PROPS

_PROP_LABELS = {
    'formation_energy':             ('Formation energy',    'eV/atom'),
    'bandgap':                      ('Band gap (indirect)', 'eV'),
    'bandgap_direct':               ('Band gap (direct)',   'eV'),
    'vbm':                          ('VBM',                 'eV'),
    'cbm':                          ('CBM',                 'eV'),
    'is_metal':                     ('Is metal',            ''),
    'effective_mass_e':             ('Eff. mass (e⁻)',      'mₑ'),
    'effective_mass_h':             ('Eff. mass (h⁺)',      'mₑ'),
    'n_sites':                      ('# sites',             ''),
    'n_species':                    ('# species',           ''),
    'volume':                       ('Volume',              'Å³'),
    'density':                      ('Density',             'g/cm³'),
    'spacegroup':                   ('Space group',         ''),
    'spacegroup_number':            ('Space group #',       ''),
    'total_energy':                 ('Total energy',        'eV'),
    'e_above_hull':                 ('E above hull',        'eV/atom'),
    # ALIGNN Tc predictions
    'tc':                           ('Critical temp (Tc)',  'K'),
    'tc_formation_energy':          ('Formation E (ALIGNN)','eV/atom'),
    'tc_bandgap':                   ('Band gap (ALIGNN)',   'eV'),
    'tc_debye_temperature':         ('Debye temp (ALIGNN)', 'K'),
    'is_superconductor_candidate':  ('SC candidate',        ''),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decode_slices(slices_str: str):
    """Return a pymatgen Structure from a SLICES string, or exit on failure."""
    from slices_ops import SLICESMutator
    mutator = SLICESMutator(element_set='oxides')  # element_set only needed for mutations
    structure = mutator.to_structure(slices_str)
    if structure is None:
        print(f"ERROR: Failed to decode SLICES string to crystal structure.")
        print(f"       Check that the string is valid (atom symbols, edge indices, PBC tokens).")
        sys.exit(1)
    return structure


def _structural_props(structure) -> dict:
    """Compute structure-derived properties via pymatgen (no DFT)."""
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    analyzer = SpacegroupAnalyzer(structure, symprec=0.1)
    sg_symbol = analyzer.get_space_group_symbol()
    sg_number = analyzer.get_space_group_number()

    return {
        'n_sites':           len(structure),
        'n_species':         len(set(str(s) for s in structure.species)),
        'volume':            round(structure.volume, 4),
        'density':           round(structure.density, 4),
        'spacegroup':        f"{sg_symbol} ({sg_number})",
        'spacegroup_number': sg_number,
    }


def _calc_chgnet(structure, relax: bool, verbose: bool) -> dict:
    """Predict formation energy and optionally relax via CHGNet."""
    try:
        from chgnet.model import CHGNet
    except ImportError:
        print("ERROR: chgnet not installed. Run: pip install chgnet")
        sys.exit(1)

    if verbose:
        print("Loading CHGNet model...")
    model = CHGNet.load()

    if relax:
        try:
            from chgnet.model.dynamics import StructOptimizer
            if verbose:
                print("Relaxing structure with CHGNet...")
            relaxer = StructOptimizer(model=model)
            result = relaxer.relax(structure, verbose=verbose)
            structure = result['final_structure']
            if verbose:
                print(f"  Relaxation converged in {result.get('steps', '?')} steps.")
        except Exception as exc:
            if verbose:
                print(f"  Relaxation failed ({exc}), using as-decoded structure.")

    if verbose:
        print("Predicting properties with CHGNet...")
    pred = model.predict_structure(structure)
    e_per_atom = float(pred['e'])

    props = _structural_props(structure)
    props['formation_energy'] = round(e_per_atom, 6)
    return props, structure


def _calc_qe(slices_str: str, structure, pseudo_dir: str | None,
             ecutwfc: float, kpoints: tuple, verbose: bool) -> dict:
    """Run Quantum ESPRESSO SCF/NSCF via CrystalQEInterface."""
    try:
        from crystal_qe_interface import CrystalQEInterface
    except ImportError:
        print("ERROR: crystal_qe_interface not found. Ensure molev_utils is on PYTHONPATH.")
        sys.exit(1)

    kpts = tuple(int(k) for k in kpoints)
    iface = CrystalQEInterface(
        pseudo_dir=pseudo_dir,
        ecutwfc=ecutwfc,
        kpoints_scf=kpts,
        kpoints_nscf=tuple(k * 2 for k in kpts),
        verbose=verbose,
    )
    props = iface.calculate(slices_str, structure=structure)
    if props.get('error'):
        print(f"ERROR from QE calculation: {props['error']}")
        sys.exit(1)
    return props, structure


def _calc_alignn_tc(slices_str: str, structure, tc_threshold: float,
                    verbose: bool) -> tuple:
    """Predict Tc and related properties via ALIGNN (InvDesFlow/JARVIS).

    Uses four pretrained ALIGNN models:
      - jv_supercon_tc_alignn        → critical temperature (K)
      - jv_formation_energy_peratom_alignn → formation energy (eV/atom)
      - jv_mbj_bandgap               → band gap (eV, metallic if < 0.05)
      - jv_supercon_debye_alignn     → Debye temperature (K)

    Screening (Wines et al. 2023): Tc > threshold AND Eform < 0 AND gap < 0.05
    """
    try:
        from crystal_tc_interface import CrystalTcInterface
    except ImportError:
        print("ERROR: crystal_tc_interface not found. Ensure molev_utils is on PYTHONPATH.")
        sys.exit(1)

    iface = CrystalTcInterface(tc_threshold=tc_threshold, verbose=verbose)
    props = iface.calculate(slices_str, structure=structure)
    if props.get('error'):
        print(f"ERROR from ALIGNN Tc prediction: {props['error']}")
        sys.exit(1)
    return props, structure


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def _print_header(slices_str: str, calculator: str):
    width = 60
    print("═" * width)
    print(f"  Crystal Property Calculator  │  Calculator: {calculator.upper()}")
    print("═" * width)
    # Truncate long SLICES strings for display
    display = slices_str if len(slices_str) <= 50 else slices_str[:47] + "..."
    print(f"  {'SLICES':<22}│ {display}")
    print("─" * width)


def _print_props(props: dict, requested: list[str]):
    width = 60
    for key in requested:
        if key not in _PROP_LABELS:
            continue
        val = props.get(key)
        label, unit = _PROP_LABELS[key]
        if val is None:
            val_str = "N/A"
        elif isinstance(val, bool):
            val_str = "yes" if val else "no"
        elif isinstance(val, float):
            val_str = f"{val:.4f}"
            if unit:
                val_str += f"  {unit}"
        elif isinstance(val, int):
            val_str = str(val)
            if unit:
                val_str += f"  {unit}"
        else:
            val_str = str(val)
        print(f"  {label:<22}│ {val_str}")
    print("═" * width)


def _print_all_props(props: dict, requested: list[str]):
    _print_props(props, requested)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Standalone crystal property calculator for SLICES genotypes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        '--calculator', choices=['chgnet', 'qe', 'alignn_tc'], default='chgnet',
        help=(
            "Property calculator to use (default: chgnet). "
            "chgnet: fast ML (~1 s, formation energy + structure). "
            "qe: accurate DFT via Quantum ESPRESSO (~minutes, band gap + eff. masses). "
            "alignn_tc: ALIGNN superconductor Tc prediction (~1 s, requires alignn + jarvis-tools)."
        ),
    )
    p.add_argument(
        '--slices', required=True, metavar='STRING',
        help="SLICES crystal genotype string, e.g. \"Ti O O  0 1 ooo  0 2 +oo\".",
    )
    p.add_argument(
        '--properties', nargs='+',
        default=['formation_energy', 'n_sites', 'n_species', 'volume', 'density', 'spacegroup'],
        metavar='PROP',
        help="Properties to compute and display. Available: " + ', '.join(sorted(_ALL_PROPS)),
    )
    p.add_argument(
        '--relax', action='store_true',
        help="(CHGNet only) Relax the structure before evaluation.",
    )
    # alignn_tc-specific
    p.add_argument('--tc-threshold', type=float, default=5.0,
                   help="(alignn_tc only) Minimum Tc (K) for superconductor candidate screening (default: 5.0).")
    # QE-specific
    p.add_argument('--pseudo-dir', metavar='PATH', default=None,
                   help="(QE only) Directory containing UPF pseudopotentials.")
    p.add_argument('--ecutwfc', type=float, default=60.0,
                   help="(QE only) Plane-wave cutoff in Ry (default: 60).")
    p.add_argument('--kpoints', nargs=3, type=int, default=[4, 4, 4],
                   metavar=('NK1', 'NK2', 'NK3'),
                   help="(QE only) Monkhorst-Pack k-grid for SCF (default: 4 4 4).")
    p.add_argument('--verbose', '-v', action='store_true',
                   help="Show progress messages.")
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    slices_str = args.slices.strip()
    calculator = args.calculator
    requested = args.properties

    # Validate requested properties
    unknown = [p for p in requested if p not in _ALL_PROPS]
    if unknown:
        print(f"WARNING: Unknown properties ignored: {', '.join(unknown)}")
        requested = [p for p in requested if p in _ALL_PROPS]

    if args.verbose:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Decoding SLICES string...")

    structure = _decode_slices(slices_str)

    if args.verbose:
        print(f"  Decoded: {len(structure)} sites, "
              f"{len(set(str(s) for s in structure.species))} species, "
              f"volume={structure.volume:.2f} Å³")

    if calculator == 'chgnet':
        props, structure = _calc_chgnet(structure, relax=args.relax, verbose=args.verbose)
    elif calculator == 'qe':
        props, structure = _calc_qe(
            slices_str, structure,
            pseudo_dir=args.pseudo_dir,
            ecutwfc=args.ecutwfc,
            kpoints=args.kpoints,
            verbose=args.verbose,
        )
    else:  # alignn_tc
        props, structure = _calc_alignn_tc(
            slices_str, structure,
            tc_threshold=args.tc_threshold,
            verbose=args.verbose,
        )
        # Default to Tc-relevant properties when user hasn't overridden
        if args.properties == ['formation_energy', 'n_sites', 'n_species',
                                'volume', 'density', 'spacegroup']:
            requested = ['tc', 'tc_formation_energy', 'tc_bandgap',
                         'tc_debye_temperature', 'is_superconductor_candidate',
                         'n_sites', 'n_species', 'spacegroup']

    _print_header(slices_str, calculator)
    _print_all_props(props, requested)


if __name__ == '__main__':
    main()
