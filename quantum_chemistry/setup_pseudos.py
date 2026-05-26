#!/usr/bin/env python3
"""
Download SSSP-efficiency pseudopotentials for Quantum ESPRESSO.

Downloads the SSSP 1.3.0 PBE-efficiency tarball from Materials Cloud and
extracts UPF files for the requested elements into a local directory.

Usage
-----
# Download pseudopotentials for all elements in the semiconductors set:
python quantum_chemistry/setup_pseudos.py --elements Si Ge Ga As N P Al In Sb

# Download for a specific run (auto-detect from a SLICES structure):
python quantum_chemistry/setup_pseudos.py --element-set semiconductors

# Specify a custom output directory:
python quantum_chemistry/setup_pseudos.py --elements Si --pseudo-dir /data/pseudos

# List pseudopotentials already present:
python quantum_chemistry/setup_pseudos.py --list

After downloading, either:
  export ESPRESSO_PSEUDO=/path/to/pseudo_dir
or pass pseudo_dir= to CrystalQEInterface / QECrystalCalculator.
"""

import argparse
import sys
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Element sets (mirrors slices_ops.py ELEMENT_SETS)
# ---------------------------------------------------------------------------
ELEMENT_SETS = {
    'oxides': [
        'Li', 'Na', 'K', 'Ca', 'Mg', 'Al', 'Si', 'Ti', 'V', 'Cr', 'Mn',
        'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Sn',
        'Ba', 'La', 'Ce', 'W', 'Pb', 'Bi', 'O',
    ],
    'semiconductor': [
        'Si', 'Ge', 'C', 'Sn', 'Ga', 'As', 'In', 'P', 'Al', 'Sb',
        'Cd', 'S', 'Se', 'Te', 'Zn', 'Hg', 'N',
    ],
    'semiconductors': [
        'C', 'Si', 'Ge', 'Sn',
        'B', 'Al', 'Ga', 'In',
        'N', 'P', 'As', 'Sb', 'Bi',
        'Zn', 'Cd', 'Hg',
        'O', 'S', 'Se', 'Te',
        'Mo', 'W',
        'Cu', 'Ag',
        'Sc',
    ],
    'halide_perovskite': [
        'Cs', 'Rb', 'K', 'Na', 'Li', 'Pb', 'Sn', 'Ge', 'I', 'Br', 'Cl',
    ],
}


def main():
    parser = argparse.ArgumentParser(
        description='Download SSSP pseudopotentials for Quantum ESPRESSO',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--elements', nargs='+', metavar='ELEM',
                        help='Element symbols to download (e.g. Si Ge Ga)')
    parser.add_argument('--element-set', choices=list(ELEMENT_SETS.keys()),
                        help='Download all elements in a predefined set')
    parser.add_argument('--pseudo-dir', default=None,
                        help='Output directory (default: ESPRESSO_PSEUDO env var '
                             'or ~/.local/share/espresso/pseudo)')
    parser.add_argument('--functional', default='PBE', choices=['PBE', 'PBEsol'],
                        help='DFT functional (default: PBE)')
    parser.add_argument('--list', action='store_true',
                        help='List pseudopotentials already present and exit')
    args = parser.parse_args()

    # Resolve pseudo directory
    _repo_root = Path(__file__).parent.parent
    sys.path.insert(0, str(_repo_root / 'quantum_chemistry' / 'calculators'))
    from qe_crystal import get_default_pseudo_dir, find_pseudo, download_sssp_pseudos

    pseudo_dir = Path(args.pseudo_dir) if args.pseudo_dir else get_default_pseudo_dir()
    pseudo_dir.mkdir(parents=True, exist_ok=True)

    # --list mode
    if args.list:
        print(f"Pseudopotential directory: {pseudo_dir}")
        upf_files = sorted(pseudo_dir.glob('*.UPF')) + sorted(pseudo_dir.glob('*.upf'))
        if upf_files:
            print(f"{len(upf_files)} UPF files found:")
            for f in upf_files:
                print(f"  {f.name}")
        else:
            print("No UPF files found.")
        return 0

    # Resolve element list
    elements = []
    if args.element_set:
        elements = ELEMENT_SETS[args.element_set]
        print(f"Element set '{args.element_set}': {len(elements)} elements")
    if args.elements:
        elements = list(dict.fromkeys(elements + args.elements))  # deduplicate

    if not elements:
        parser.error("Specify --elements or --element-set")

    print(f"Downloading {args.functional} SSSP pseudopotentials for: "
          f"{', '.join(elements)}")
    print(f"Output directory: {pseudo_dir}")
    print()

    try:
        result = download_sssp_pseudos(elements, pseudo_dir, functional=args.functional)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    ok = sum(1 for v in result.values() if v is not None)
    fail = [e for e, v in result.items() if v is None]
    print(f"\n{ok}/{len(elements)} pseudopotentials ready.")
    if fail:
        print(f"Missing: {', '.join(fail)}")
        print("These elements may not be in the SSSP library for this functional.")
        return 1

    print(f"\nSet ESPRESSO_PSEUDO={pseudo_dir} before running QE calculations,")
    print("or pass pseudo_dir='{}' to CrystalQEInterface.".format(pseudo_dir))
    return 0


if __name__ == '__main__':
    sys.exit(main())
