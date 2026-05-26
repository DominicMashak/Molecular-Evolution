"""
Quantum ESPRESSO DFT calculator for periodic crystal structures.

Three-step pw.x workflow:
  1. SCF   — converged charge density + total energy (eV/atom)
  2. NSCF  — dense k-mesh eigenvalues → direct/indirect bandgap, VBM/CBM
  3. Bands — E(k) along high-symmetry path → electron/hole effective masses

Formation energy is computed vs pre-tabulated PBE elemental reference energies
(Materials Project GGA-PBE).  For quantitative accuracy recompute references
with your own pseudopotentials using calc_elemental_reference().

Prerequisites
-------------
  pw.x in PATH (or QE_BINARY env var):
    conda install -n mol-evo -c conda-forge qe

  Pseudopotentials — SSSP-efficiency UPF files, one per element:
    from quantum_chemistry.calculators.qe_crystal import download_sssp_pseudos
    download_sssp_pseudos(['Si','Ge','GaAs'], '/path/to/pseudo')

  Or set ESPRESSO_PSEUDO environment variable to an existing pseudo directory.

Usage
-----
    from quantum_chemistry.calculators.qe_crystal import QECrystalCalculator
    from pymatgen.core import Structure, Lattice

    si = Structure(Lattice.cubic(5.43), ['Si','Si'],
                   [[0,0,0],[0.25,0.25,0.25]])
    calc = QECrystalCalculator(pseudo_dir='/path/to/pseudo')
    results = calc.calculate(si)
    print(results['bandgap'], results['effective_mass_e'])
"""

from __future__ import annotations

import logging
import os
import re
import sys
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
_HA_TO_EV: float = 27.211386          # 1 Hartree = 27.211386 eV
_HBAR2_2ME: float = 3.80998           # ℏ²/(2mₑ) in eV·Å²  (used for eff. mass)
_BOHR_TO_ANG: float = 0.529177        # 1 Bohr = 0.529177 Å
_HBAR_KB: float = 7.6382e-12          # ℏ/k_B in s·K (Debye temperature)
_KB_EV: float    = 8.617333e-5        # k_B in eV/K
_M0_KG: float    = 9.109384e-31       # electron rest mass in kg
_E_C: float      = 1.602176e-19       # elementary charge in C
_HBAR_J: float   = 1.054572e-34       # ℏ in J·s

# ---------------------------------------------------------------------------
# PBE elemental reference energies (eV/atom, most stable polymorph)
# Source: Materials Project GGA-PBE.  These are indicative values — recompute
# with your pseudopotentials + cutoffs for quantitative formation energies.
# ---------------------------------------------------------------------------
PBE_ELEMENTAL_REFERENCES: Dict[str, float] = {
    'H':  -3.381,   # ½ H₂
    'C':  -9.226,   # diamond
    'Si': -5.425,   # diamond cubic
    'Ge': -4.623,   # diamond cubic
    'Sn': -3.985,   # white tin (α)
    'B':  -6.678,   # β-rhombohedral
    'Al': -3.748,   # FCC
    'Ga': -3.031,   # orthorhombic α-Ga
    'In': -2.721,   # body-centred tetragonal
    'N':  -8.340,   # ½ N₂
    'P':  -5.159,   # black phosphorus
    'As': -4.654,   # grey arsenic (rhombohedral)
    'Sb': -4.118,   # rhombohedral α-Sb
    'Bi': -3.871,   # rhombohedral
    'O':  -4.948,   # ½ O₂  (note: GGA O₂ binding energy error ~0.7 eV)
    'S':  -4.119,   # α-S (orthorhombic)
    'Se': -3.498,   # trigonal
    'Te': -3.143,   # trigonal
    'Zn': -1.266,   # hexagonal
    'Cd': -0.908,   # hexagonal
    'Hg': -0.300,   # rhombohedral (extrapolated)
    'Mo': -10.850,  # BCC
    'W':  -12.960,  # BCC
    'Cu': -3.749,   # FCC
    'Ag': -2.827,   # FCC
    'Sc': -6.338,   # HCP
}

# Approximate valence electron counts for SSSP-efficiency pseudopotentials.
# The exact values are in each UPF file's z_valence field; these defaults
# are used only when the UPF file cannot be read.
_DEFAULT_VALENCE: Dict[str, int] = {
    'H': 1, 'He': 2, 'Li': 1, 'Be': 2, 'B': 3, 'C': 4, 'N': 5, 'O': 6,
    'F': 7, 'Na': 1, 'Mg': 2, 'Al': 3, 'Si': 4, 'P': 5, 'S': 6, 'Cl': 7,
    'K': 1, 'Ca': 2, 'Sc': 11, 'Ti': 12, 'V': 13, 'Cr': 14, 'Mn': 7,
    'Fe': 16, 'Co': 17, 'Ni': 18, 'Cu': 11, 'Zn': 12, 'Ga': 13, 'Ge': 4,
    'As': 5, 'Se': 6, 'Br': 7, 'Kr': 8, 'Rb': 1, 'Sr': 2, 'Y': 11,
    'Zr': 12, 'Nb': 13, 'Mo': 14, 'Ru': 16, 'Rh': 17, 'Pd': 10, 'Ag': 11,
    'Cd': 12, 'In': 13, 'Sn': 4, 'Sb': 5, 'Te': 6, 'I': 7, 'Cs': 1,
    'Ba': 2, 'Hf': 12, 'Ta': 13, 'W': 14, 'Os': 16, 'Ir': 17, 'Pt': 10,
    'Au': 11, 'Hg': 12, 'Tl': 3, 'Pb': 4, 'Bi': 5,
}

# ---------------------------------------------------------------------------
# Pseudopotential utilities
# ---------------------------------------------------------------------------

def get_default_pseudo_dir() -> Path:
    """Return the default pseudopotential directory.

    Checks (in order):
      1. ESPRESSO_PSEUDO environment variable
      2. ~/.local/share/espresso/pseudo/
    """
    env = os.environ.get('ESPRESSO_PSEUDO')
    if env:
        p = Path(env)
        if p.is_dir():
            return p
    default = Path.home() / '.local' / 'share' / 'espresso' / 'pseudo'
    default.mkdir(parents=True, exist_ok=True)
    return default


def find_pseudo(element: str, pseudo_dir: Path) -> Optional[Path]:
    """Find a UPF pseudopotential file for *element* in *pseudo_dir*.

    Searches case-insensitively for ``{Element}.*.upf``.  Returns the first
    match, or None if none found.
    """
    pseudo_dir = Path(pseudo_dir)
    pattern_lower = element.lower()
    for p in sorted(pseudo_dir.iterdir()):
        name = p.name.lower()
        if name.startswith(pattern_lower + '.') and name.endswith('.upf'):
            return p
    return None


def read_upf_valence(upf_file: Path) -> Optional[int]:
    """Read the number of valence electrons from a UPF pseudopotential file.

    Supports both UPF v1 (plain text) and UPF v2 (XML-like) formats.
    """
    try:
        text = upf_file.read_text(errors='replace')
        # UPF v2 XML: <z_valence>4.000000</z_valence>
        m = re.search(r'<z_valence[^>]*>\s*([\d.]+)\s*</z_valence>', text)
        if m:
            return round(float(m.group(1)))
        # UPF v1 plaintext: "   Z valence            4.000000000"
        m = re.search(r'Z\s+valence\s+([\d.]+)', text, re.IGNORECASE)
        if m:
            return round(float(m.group(1)))
    except Exception:
        pass
    return None


def download_sssp_pseudos(elements: List[str],
                           pseudo_dir: Optional[str] = None,
                           functional: str = 'PBE') -> Dict[str, Path]:
    """Download SSSP-efficiency pseudopotentials from the QE pseudopotential repository.

    Downloads individual UPF files per element from the Quantum ESPRESSO
    pseudopotential repository.  Skips elements already present in *pseudo_dir*.

    Args:
        elements:    List of element symbols (e.g. ['Si', 'Ge', 'Ga', 'As']).
        pseudo_dir:  Directory to store UPF files (default: ~/.local/share/
                     espresso/pseudo/).
        functional:  'PBE' (default) or 'PBEsol'.

    Returns:
        Dict mapping element → Path of the downloaded UPF file (None if failed).
    """
    import urllib.request

    pseudo_dir = Path(pseudo_dir or get_default_pseudo_dir())
    pseudo_dir.mkdir(parents=True, exist_ok=True)

    # SSSP 1.3.0 efficiency filenames from Quantum ESPRESSO pseudopotential library.
    # Source: https://pseudopotentials.quantum-espresso.org/legacy_tables/sssp
    # Format: element → (PBE_filename, PBEsol_filename)
    _SSSP_FILES: Dict[str, Tuple[str, str]] = {
        'H':  ('H.pbe-rrkjus_psl.1.0.0.UPF',       'H.pbesol-rrkjus_psl.1.0.0.UPF'),
        'He': ('He.pbe-mt_fhi.UPF',                  'He.pbesol-mt_fhi.UPF'),
        'Li': ('li_pbe_v1.4.uspp.F.UPF',             'li_pbesol_v1.4.uspp.F.UPF'),
        'Be': ('be_pbe_v1.4.uspp.F.UPF',             'be_pbesol_v1.4.uspp.F.UPF'),
        'B':  ('B.pbe-n-kjpaw_psl.1.0.0.UPF',          'B.pbesol-n-kjpaw_psl.1.0.0.UPF'),
        'C':  ('C.pbe-n-kjpaw_psl.1.0.0.UPF',        'C.pbesol-n-kjpaw_psl.1.0.0.UPF'),
        'N':  ('N.pbe-n-kjpaw_psl.1.0.0.UPF',          'N.pbesol-n-kjpaw_psl.1.0.0.UPF'),
        'O':  ('O.pbe-n-kjpaw_psl.0.1.UPF',          'O.pbesol-n-kjpaw_psl.0.1.UPF'),
        'F':  ('F.pbe-n-kjpaw_psl.1.0.0.UPF',        'F.pbesol-n-kjpaw_psl.1.0.0.UPF'),
        'Na': ('Na.pbe-spn-kjpaw_psl.1.0.0.UPF',     'Na.pbesol-spn-kjpaw_psl.1.0.0.UPF'),
        'Mg': ('Mg.pbe-spnl-kjpaw_psl.1.0.0.UPF',    'Mg.pbesol-spnl-kjpaw_psl.1.0.0.UPF'),
        'Al': ('Al.pbe-n-kjpaw_psl.1.0.0.UPF',       'Al.pbesol-n-kjpaw_psl.1.0.0.UPF'),
        'Si': ('Si.pbe-n-rrkjus_psl.1.0.0.UPF',      'Si.pbesol-n-rrkjus_psl.1.0.0.UPF'),
        'P':  ('P.pbe-n-rrkjus_psl.1.0.0.UPF',       'P.pbesol-n-rrkjus_psl.1.0.0.UPF'),
        'S':  ('S.pbe-n-rrkjus_psl.1.0.0.UPF',       'S.pbesol-n-rrkjus_psl.1.0.0.UPF'),
        'Cl': ('Cl.pbe-n-rrkjus_psl.1.0.0.UPF',      'Cl.pbesol-n-rrkjus_psl.1.0.0.UPF'),
        'K':  ('K.pbe-spn-kjpaw_psl.1.0.0.UPF',      'K.pbesol-spn-kjpaw_psl.1.0.0.UPF'),
        'Ca': ('Ca.pbe-spn-kjpaw_psl.1.0.0.UPF',     'Ca.pbesol-spn-kjpaw_psl.1.0.0.UPF'),
        'Sc': ('Sc.pbe-spn-kjpaw_psl.1.0.0.UPF',     'Sc.pbesol-spn-kjpaw_psl.1.0.0.UPF'),
        'Ti': ('ti_pbe_v1.4.uspp.F.UPF',             'ti_pbesol_v1.4.uspp.F.UPF'),
        'V':  ('v_pbe_v1.4.uspp.F.UPF',              'v_pbesol_v1.4.uspp.F.UPF'),
        'Cr': ('cr_pbe_v1.5.uspp.F.UPF',             'cr_pbesol_v1.5.uspp.F.UPF'),
        'Mn': ('mn_pbe_v1.5.uspp.F.UPF',             'mn_pbesol_v1.5.uspp.F.UPF'),
        'Fe': ('fe_pbe_v1.5.uspp.F.UPF',             'fe_pbesol_v1.5.uspp.F.UPF'),
        'Co': ('co_pbe_v1.2.uspp.F.UPF',             'co_pbesol_v1.2.uspp.F.UPF'),
        'Ni': ('ni_pbe_v1.4.uspp.F.UPF',             'ni_pbesol_v1.4.uspp.F.UPF'),
        'Cu': ('Cu.pbe-dn-rrkjus_psl.1.0.0.UPF',     'Cu.pbesol-dn-rrkjus_psl.1.0.0.UPF'),
        'Zn': ('Zn.pbe-d-hgh.UPF',                   'Zn.pbe-d-hgh.UPF'),
        'Ga': ('Ga.pbe-dn-kjpaw_psl.1.0.0.UPF',     'Ga.pbesol-dn-kjpaw_psl.1.0.0.UPF'),
        'Ge': ('Ge.pbe-dn-kjpaw_psl.1.0.0.UPF',     'Ge.pbesol-dn-kjpaw_psl.1.0.0.UPF'),
        'As': ('As.pbe-n-rrkjus_psl.1.0.0.UPF',     'As.pbesol-n-rrkjus_psl.1.0.0.UPF'),
        'Se': ('Se.pbe-dn-rrkjus_psl.1.0.0.UPF',    'Se.pbesol-dn-rrkjus_psl.1.0.0.UPF'),
        'Br': ('Br.pbe-dn-rrkjus_psl.1.0.0.UPF',    'Br.pbesol-dn-rrkjus_psl.1.0.0.UPF'),
        'Rb': ('Rb.pbe-spn-kjpaw_psl.1.0.0.UPF',    'Rb.pbesol-spn-kjpaw_psl.1.0.0.UPF'),
        'Sr': ('Sr.pbe-spn-kjpaw_psl.1.0.0.UPF',    'Sr.pbesol-spn-kjpaw_psl.1.0.0.UPF'),
        'Y':  ('Y.pbe-spn-kjpaw_psl.1.0.0.UPF',     'Y.pbesol-spn-kjpaw_psl.1.0.0.UPF'),
        'Zr': ('Zr.pbe-spn-kjpaw_psl.1.0.0.UPF',    'Zr.pbesol-spn-kjpaw_psl.1.0.0.UPF'),
        'Nb': ('Nb.pbe-spn-kjpaw_psl.1.0.0.UPF',    'Nb.pbesol-spn-kjpaw_psl.1.0.0.UPF'),
        'Mo': ('Mo.pbe-spn-kjpaw_psl.1.0.0.UPF',    'Mo.pbesol-spn-kjpaw_psl.1.0.0.UPF'),
        'Sn': ('Sn.pbe-dn-rrkjus_psl.1.0.0.UPF',    'Sn.pbesol-dn-rrkjus_psl.1.0.0.UPF'),
        'Sb': ('Sb.pbe-n-rrkjus_psl.1.0.0.UPF',     'Sb.pbesol-n-rrkjus_psl.1.0.0.UPF'),
        'Te': ('Te.pbe-n-rrkjus_psl.1.0.0.UPF',     'Te.pbesol-n-rrkjus_psl.1.0.0.UPF'),
        'I':  ('I.pbe-dn-rrkjus_psl.1.0.0.UPF',     'I.pbesol-dn-rrkjus_psl.1.0.0.UPF'),
        'Cs': ('Cs.pbe-spn-kjpaw_psl.1.0.0.UPF',    'Cs.pbesol-spn-kjpaw_psl.1.0.0.UPF'),
        'Ba': ('Ba.pbe-spn-kjpaw_psl.1.0.0.UPF',    'Ba.pbesol-spn-kjpaw_psl.1.0.0.UPF'),
        'La': ('La.pbe-spfn-kjpaw_psl.1.0.0.UPF',   'La.pbesol-spfn-kjpaw_psl.1.0.0.UPF'),
        'Ce': ('Ce.GGA-PBE-paw-v1.0.UPF',            'Ce.GGA-PBEsol-paw-v1.0.UPF'),
        'Hg': ('Hg.pbe-spn-kjpaw_psl.1.0.0.UPF',     'Hg.pbe-spn-kjpaw_psl.1.0.0.UPF'),
        'Pb': ('Pb.pbe-dn-kjpaw_psl.1.0.0.UPF',     'Pb.pbesol-dn-kjpaw_psl.1.0.0.UPF'),
        'Bi': ('Bi.pbe-dn-kjpaw_psl.1.0.0.UPF',     'Bi.pbesol-dn-kjpaw_psl.1.0.0.UPF'),
        'W':  ('W.pbe-spn-kjpaw_psl.1.0.0.UPF',     'W.pbesol-spn-kjpaw_psl.1.0.0.UPF'),
        'In': ('In.pbe-dn-rrkjus_psl.1.0.0.UPF',    'In.pbesol-dn-rrkjus_psl.1.0.0.UPF'),
        'Ag': ('Ag.pbe-n-rrkjus_psl.1.0.0.UPF',      'Ag.pbe-n-rrkjus_psl.1.0.0.UPF'),
        'Cd': ('Cd.pbe-dn-kjpaw_psl.0.3.1.UPF',      'Cd.pbe-dn-kjpaw_psl.0.3.1.UPF'),
    }

    # Base URL for the QE pseudopotential repository
    _BASE_URL = 'https://pseudopotentials.quantum-espresso.org/upf_files/'

    func_upper = functional.upper()
    if func_upper not in ('PBE', 'PBESOL'):
        func_upper = 'PBE'
    func_idx = 0 if func_upper == 'PBE' else 1

    # Check which elements are missing
    missing = [e for e in elements if find_pseudo(e, pseudo_dir) is None]
    if not missing:
        logger.info("All pseudopotentials already present.")
        return {e: find_pseudo(e, pseudo_dir) for e in elements}

    logger.info(f"Downloading SSSP {functional} efficiency pseudopotentials for: "
                f"{', '.join(missing)}")

    result: Dict[str, Optional[Path]] = {}
    for elem in missing:
        entry = _SSSP_FILES.get(elem)
        if entry is None:
            logger.warning(f"  {elem}: no SSSP filename mapping — skipping")
            result[elem] = None
            continue

        filename = entry[func_idx]
        dest = pseudo_dir / filename
        url = _BASE_URL + filename

        if dest.exists():
            logger.info(f"  {elem}: already present ({filename})")
            result[elem] = dest
            continue

        try:
            logger.info(f"  {elem}: downloading {filename} …")
            urllib.request.urlretrieve(url, dest)
            logger.info(f"  {elem}: saved to {dest}")
            result[elem] = dest
        except Exception as exc:
            logger.error(f"  {elem}: download failed — {exc}")
            result[elem] = None

    # Add already-present elements
    for elem in elements:
        if elem not in result:
            result[elem] = find_pseudo(elem, pseudo_dir)

    return result


# ---------------------------------------------------------------------------
# QECrystalCalculator
# ---------------------------------------------------------------------------

class QECrystalCalculator:
    """Quantum ESPRESSO DFT calculator for periodic crystal structures.

    Runs three pw.x calculations: SCF → NSCF → Bands, and extracts:
      - Total energy and formation energy (eV/atom)
      - Direct and indirect electronic bandgap (eV)
      - VBM / CBM positions (eV, relative to Fermi level)
      - Electron and hole effective masses (mₑ)
      - Full band structure along high-symmetry k-path

    Args:
        functional:     XC functional name for QE ('PBE', 'PBEsol').
        ecutwfc:        Plane-wave kinetic energy cutoff (Ry).
        ecutrho:        Charge density cutoff (Ry); should be 4–8× ecutwfc.
        kpoints_scf:    Monkhorst-Pack grid for SCF, e.g. (4, 4, 4).
        kpoints_nscf:   Denser grid for NSCF bandgap, e.g. (8, 8, 8).
        nk_bands:       Approximate number of k-points along band path.
        pseudo_dir:     Directory containing UPF pseudopotentials.
                        If None, uses ESPRESSO_PSEUDO env var or
                        ~/.local/share/espresso/pseudo/.
        work_dir:       Base directory for QE work directories.  Each call
                        creates a unique sub-directory.  Defaults to /tmp.
        keep_workdir:   Keep work directories after calculation (for debug).
        conv_thr:       SCF convergence threshold (Ry).
        smearing:       Occupations smearing ('gaussian', 'mv', 'mp').
        degauss:        Smearing width (Ry).
        verbose:        Print QE stdout to terminal.
    """

    def __init__(
        self,
        functional: str = 'PBE',
        ecutwfc: float = 60.0,
        ecutrho: float = 480.0,
        kpoints_scf: Tuple[int, int, int] = (4, 4, 4),
        kpoints_nscf: Tuple[int, int, int] = (8, 8, 8),
        nk_bands: int = 100,
        pseudo_dir: Optional[str] = None,
        work_dir: Optional[str] = None,
        keep_workdir: bool = False,
        conv_thr: float = 1e-8,
        smearing: str = 'gaussian',
        degauss: float = 0.01,
        verbose: bool = False,
        calc_hardness: bool = False,
        strain_delta: float = 0.01,
        calc_phonon_stability: bool = False,
        calc_mobility: bool = False,
        calc_hull_distance: bool = False,
    ) -> None:
        self.functional = functional
        self.ecutwfc = ecutwfc
        self.ecutrho = ecutrho
        self.kpoints_scf = tuple(kpoints_scf)
        self.kpoints_nscf = tuple(kpoints_nscf)
        self.nk_bands = nk_bands
        self.pseudo_dir = Path(pseudo_dir) if pseudo_dir else get_default_pseudo_dir()
        self.work_base = Path(work_dir) if work_dir else Path(tempfile.gettempdir())
        self.keep_workdir = keep_workdir
        self.conv_thr = conv_thr
        self.smearing = smearing
        self.degauss = degauss
        self.verbose = verbose
        self.calc_hardness = calc_hardness
        # Strain amplitude for finite-difference elastic constants (engineering strain).
        # 0.01 (1%) is safe for most semiconductors; increase to 0.02 for soft materials.
        self.strain_delta = strain_delta
        # Phonon stability: single ph.x run at Γ to detect imaginary modes.
        self.calc_phonon_stability = calc_phonon_stability
        # Carrier mobility via deformation potential (+4 SCF+NSCF runs).
        self.calc_mobility = calc_mobility
        # Convex hull distance via CHGNet + pymatgen PhaseDiagram.
        self.calc_hull_distance = calc_hull_distance
        self._chgnet_model = None          # lazy-loaded CHGNet model
        self._hull_cache: dict = {}        # keyed by frozenset(element symbols)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def calculate(self, structure) -> dict:
        """Run the full QE workflow for a pymatgen Structure.

        Returns
        -------
        dict with keys:
          total_energy       (float)  eV/atom — raw DFT total energy
          formation_energy   (float)  eV/atom — vs PBE elemental refs
          bandgap            (float)  eV — indirect fundamental gap (0 = metallic)
          bandgap_direct     (float)  eV — smallest direct gap at any k-point
          vbm                (float)  eV — valence band max (Fermi-aligned)
          cbm                (float)  eV — conduction band min (Fermi-aligned)
          is_metal           (bool)
          effective_mass_e   (float)  mₑ — electron (at CBM)
          effective_mass_h   (float)  mₑ — hole (at VBM, positive)
          band_structure     (dict | None)  {'kpoints_frac', 'kpoints_cart',
                                             'energies_ev', 'labels',
                                             'label_indices', 'efermi_ev'}
          spacegroup         (int)
          n_sites            (int)
          n_species          (int)
          volume             (float)  Å³
          density            (float)  g/cm³
          error              (str | None)

        The following keys are only populated when calc_hardness=True:
          bulk_modulus       (float | None)  GPa — Voigt-Reuss-Hill average
          shear_modulus      (float | None)  GPa — Voigt-Reuss-Hill average
          youngs_modulus     (float | None)  GPa — E = 9BG/(3B+G)
          poisson_ratio      (float | None)  dimensionless — ν = (3B-2G)/(2(3B+G))
          hardness_vickers   (float | None)  GPa — Chen model: 2(k²G)^0.585 - 3
          elastic_tensor     (list | None)   6×6 list (GPa) in Voigt notation
          pugh_ratio         (float | None)  G/B — >0.57 brittle, <0.57 ductile
          cauchy_pressure    (float | None)  GPa — C₁₂-C₄₄; >0 metallic, <0 covalent
          sound_velocity_longitudinal (float | None)  m/s — v_l = √((B+4G/3)/ρ)
          sound_velocity_transverse   (float | None)  m/s — v_t = √(G/ρ)
          debye_temperature  (float | None)  K — from elastic sound velocities
          fracture_toughness (float | None)  MPa√m — Mazhnik-Oganov: 0.374√(G·H_v)

        The following keys are only populated when calc_phonon_stability=True:
          is_dynamically_stable (bool | None)  True if no imaginary phonon modes at Γ
          n_imaginary_modes     (int | None)   count of imaginary modes (ω < -10 cm⁻¹)
          phonon_frequencies_cm1 (list | None) all Γ-point frequencies in cm⁻¹
        """
        props = self._empty_props(structure)
        work = Path(tempfile.mkdtemp(prefix='qe_', dir=self.work_base))
        try:
            # Structural properties (no calculation needed)
            props.update(self._structural_props(structure))

            # Validate pseudopotentials
            pseudos, valences, err = self._resolve_pseudos(structure)
            if err:
                props['error'] = err
                return props

            # Step 1 — SCF
            scf_ok, scf_data = self._run_scf(structure, pseudos, work)
            if not scf_ok:
                props['error'] = f"SCF failed: {scf_data.get('error', 'unknown')}"
                return props
            props['total_energy'] = scf_data['total_energy_ev_per_atom']
            props['formation_energy'] = self._formation_energy(
                structure, scf_data['total_energy_ev_per_atom'])

            # Total valence electrons from pseudopotentials
            n_elec = sum(valences[s.specie.symbol] for s in structure)

            # Step 2 — NSCF (bandgap)
            nscf_ok, nscf_data = self._run_nscf(structure, pseudos, work, n_elec)
            if nscf_ok:
                props.update({
                    'bandgap':        nscf_data['gap'],
                    'bandgap_direct': nscf_data['direct_gap'],
                    'vbm':            nscf_data['vbm'],
                    'cbm':            nscf_data['cbm'],
                    'is_metal':       nscf_data['is_metal'],
                })
            else:
                props['error'] = f"NSCF failed: {nscf_data.get('error', 'unknown')}"

            # Step 3 — Bands (effective masses + band structure)
            bands_ok, bands_data = self._run_bands(structure, pseudos, work, n_elec)
            if bands_ok:
                hob = int(n_elec // 2) - 1   # 0-indexed highest occupied band
                lub = hob + 1
                em = self._effective_masses(
                    bands_data['kpoints_cart'],
                    bands_data['energies_ev'],
                    hob, lub,
                )
                props['effective_mass_e'] = em['m_electron']
                props['effective_mass_h'] = em['m_hole']
                props['band_structure'] = {
                    'kpoints_frac':   bands_data['kpoints_frac'],
                    'kpoints_cart':   bands_data['kpoints_cart'],
                    'energies_ev':    bands_data['energies_ev'],
                    'labels':         bands_data['labels'],
                    'label_indices':  bands_data['label_indices'],
                    'efermi_ev':      scf_data['efermi_ev'],
                }
            else:
                props['error'] = (props.get('error') or
                                  f"Bands failed: {bands_data.get('error', 'unknown')}")

            # Tier 2 — gap type + radiative lifetime (free; uses NSCF + Bands results)
            props['gap_type'] = self._classify_gap_type(
                props.get('bandgap', 0.0), props.get('bandgap_direct', 0.0))
            tau, B_coeff, ni = self._radiative_lifetime(
                props.get('bandgap', 0.0),
                props.get('effective_mass_e'),
                props.get('effective_mass_h'),
            )
            props['carrier_lifetime_radiative'] = tau
            props['recombination_coefficient_B'] = B_coeff
            props['intrinsic_carrier_density']   = ni

            # Step 4 (optional) — Elastic constants → Vickers hardness + derived props
            # Requires 12 additional SCF calculations (6 Voigt strains × ±δ).
            if self.calc_hardness:
                C, elastic_err = self._compute_elastic_tensor(
                    structure, pseudos, valences, work)
                if C is not None:
                    B, G, E, nu = self._voigt_reuss_hill(C)
                    H_v = self._chen_hardness(B, G)
                    props.update({
                        'bulk_modulus':     B,
                        'shear_modulus':    G,
                        'youngs_modulus':   E,
                        'poisson_ratio':    nu,
                        'hardness_vickers': H_v,
                        'elastic_tensor':   C.tolist(),
                    })
                    # Free derived properties from elastic constants
                    props.update(self._derived_elastic_props(
                        B, G, C, H_v, props['density'],
                        props['n_sites'], props['volume']))
                    # Tier 1 extras: thermal conductivity, Zener anisotropy,
                    # Peierls-Nabarro stress (all free from elastic constants)
                    nu = props.get('poisson_ratio') or 0.25
                    props.update(self._tier1_elastic_extras(
                        B, G, C, H_v, nu, props['density'],
                        props['n_sites'], props['volume']))
                else:
                    logger.warning(f"Elastic tensor calculation failed: {elastic_err}")
                    props['error'] = props.get('error') or f'Elastic: {elastic_err}'

            # Step 5 (optional) — Phonon stability at Γ via ph.x
            # Detects imaginary modes indicating structural instability.
            if self.calc_phonon_stability:
                ph_ok, ph_data = self._run_phonon_gamma(structure, pseudos, work)
                if ph_ok:
                    props['is_dynamically_stable'] = ph_data['is_stable']
                    props['n_imaginary_modes']      = ph_data['n_imaginary']
                    props['phonon_frequencies_cm1'] = ph_data['frequencies_cm1']
                else:
                    logger.warning(f"Phonon calculation failed: {ph_data.get('error', 'unknown')}")
                    props['error'] = (props.get('error') or
                                      f"Phonon: {ph_data.get('error', 'unknown')}")

            # Tier 3 (optional) — carrier mobility via deformation potential (+4 SCF+NSCF)
            if self.calc_mobility:
                dp_ok, dp_data = self._compute_deformation_potential(
                    structure, pseudos, valences, work)
                if dp_ok:
                    props['deformation_potential_e'] = dp_data['dp_e']
                    props['deformation_potential_h'] = dp_data['dp_h']
                    # Use C₁₁ from elastic tensor if available, else bulk modulus fallback
                    et = props.get('elastic_tensor')
                    c11 = et[0][0] if et else (props.get('bulk_modulus') or 100.0)
                    mu_e, mu_h = self._dp_mobility(
                        props.get('effective_mass_e'),
                        props.get('effective_mass_h'),
                        dp_data['dp_e'],
                        dp_data['dp_h'],
                        float(c11),
                    )
                    props['mobility_e'] = mu_e
                    props['mobility_h'] = mu_h
                else:
                    logger.warning(f"Deformation potential failed: "
                                   f"{dp_data.get('error', 'unknown')}")

            # Tier 4 (optional) — convex hull distance via CHGNet
            if self.calc_hull_distance:
                e_hull, is_stable = self._compute_hull_distance(structure)
                props['e_above_hull']                = e_hull
                props['is_thermodynamically_stable'] = is_stable

        finally:
            if not self.keep_workdir and work.exists():
                shutil.rmtree(work, ignore_errors=True)

        return props

    # ------------------------------------------------------------------
    # Step 1: SCF
    # ------------------------------------------------------------------

    def _run_scf(self, structure, pseudos, work) -> Tuple[bool, dict]:
        kcard = (f"K_POINTS automatic\n"
                 f"  {self.kpoints_scf[0]} {self.kpoints_scf[1]} "
                 f"{self.kpoints_scf[2]}  0 0 0\n")
        inp = self._write_pw_input(structure, pseudos, work,
                                   calculation='scf', kpoints_card=kcard)
        out = work / 'pw.scf.out'
        ok = self._run_pw_x(inp, out, work)
        if not ok:
            return False, {'error': self._tail_error(out)}

        # Parse total energy and Fermi level from XML
        xml_data = self._parse_xml(work)
        if xml_data is None:
            return False, {'error': 'Could not parse SCF XML output'}

        n_atoms = structure.num_sites
        return True, {
            'total_energy_ev_per_atom': xml_data['total_energy_ha'] * _HA_TO_EV / n_atoms,
            'efermi_ev':                xml_data['fermi_energy_ha'] * _HA_TO_EV,
        }

    # ------------------------------------------------------------------
    # Step 2: NSCF — bandgap from dense k-mesh
    # ------------------------------------------------------------------

    def _run_nscf(self, structure, pseudos, work, n_elec) -> Tuple[bool, dict]:
        # Extra bands above Fermi level
        nbnd = int(n_elec // 2) + 10
        kcard = (f"K_POINTS automatic\n"
                 f"  {self.kpoints_nscf[0]} {self.kpoints_nscf[1]} "
                 f"{self.kpoints_nscf[2]}  0 0 0\n")
        inp = self._write_pw_input(structure, pseudos, work,
                                   calculation='nscf', kpoints_card=kcard,
                                   nbnd=nbnd)
        out = work / 'pw.nscf.out'
        ok = self._run_pw_x(inp, out, work)
        if not ok:
            return False, {'error': self._tail_error(out)}

        xml_data = self._parse_xml(work)
        if xml_data is None:
            return False, {'error': 'Could not parse NSCF XML output'}

        gap_data = self._bandgap_from_eigenvalues(
            xml_data['eigenvalues_ha'],
            xml_data['fermi_energy_ha'],
            n_elec,
        )
        return True, gap_data

    # ------------------------------------------------------------------
    # Step 3: Bands — high-symmetry path
    # ------------------------------------------------------------------

    def _run_bands(self, structure, pseudos, work, n_elec) -> Tuple[bool, dict]:
        try:
            from pymatgen.symmetry.bandstructure import HighSymmKpath
            from pymatgen.core.lattice import Lattice as PMGLattice
        except ImportError:
            return False, {'error': 'pymatgen not installed'}

        # Generate high-symmetry k-path (fractional reciprocal coords)
        try:
            kpath = HighSymmKpath(structure)
            kpts_frac, labels = kpath.get_kpoints(line_density=self.nk_bands // 10)
        except Exception as e:
            return False, {'error': f'k-path generation failed: {e}'}

        n_kpts = len(kpts_frac)
        nbnd = int(n_elec // 2) + 10
        weight = 1.0 / n_kpts

        kcard_lines = [f"K_POINTS crystal\n  {n_kpts}\n"]
        for kpt in kpts_frac:
            kcard_lines.append(f"  {kpt[0]:.10f}  {kpt[1]:.10f}  {kpt[2]:.10f}"
                               f"  {weight:.10f}\n")
        kcard = ''.join(kcard_lines)

        inp = self._write_pw_input(structure, pseudos, work,
                                   calculation='bands', kpoints_card=kcard,
                                   nbnd=nbnd)
        out = work / 'pw.bands.out'
        ok = self._run_pw_x(inp, out, work)
        if not ok:
            return False, {'error': self._tail_error(out)}

        xml_data = self._parse_xml(work)
        if xml_data is None or 'eigenvalues_ha' not in xml_data:
            return False, {'error': 'Could not parse bands XML output'}

        # Convert eigenvalues and k-points
        eigenvalues_ev = np.array(xml_data['eigenvalues_ha']) * _HA_TO_EV
        efermi_ev = xml_data['fermi_energy_ha'] * _HA_TO_EV
        eigenvalues_ev -= efermi_ev   # Fermi-align

        # Convert fractional k-points to Cartesian (Å⁻¹) using reciprocal lattice
        rec_lat = structure.lattice.reciprocal_lattice.matrix  # Å⁻¹
        kpts_cart = np.array(kpts_frac) @ rec_lat              # (n_k, 3) in Å⁻¹

        # Identify high-symmetry labels and their indices
        label_indices = []
        label_symbols = []
        for i, lbl in enumerate(labels):
            if lbl:
                label_indices.append(i)
                label_symbols.append(lbl)

        return True, {
            'kpoints_frac':  kpts_frac,
            'kpoints_cart':  kpts_cart.tolist(),
            'energies_ev':   eigenvalues_ev.tolist(),
            'labels':        label_symbols,
            'label_indices': label_indices,
        }

    # ------------------------------------------------------------------
    # Bandgap extraction from k-mesh eigenvalues
    # ------------------------------------------------------------------

    def _bandgap_from_eigenvalues(self, eigenvalues_ha, fermi_ha, n_elec) -> dict:
        """Compute direct and indirect bandgap from NSCF eigenvalues.

        Args:
            eigenvalues_ha: list of shape [n_kpoints][n_bands] (Hartree)
            fermi_ha:       Fermi energy (Hartree)
            n_elec:         total number of valence electrons

        Returns dict with: gap, direct_gap, vbm, cbm, is_metal (eV)
        """
        eigs = np.array(eigenvalues_ha) * _HA_TO_EV   # (n_k, n_bands) eV
        ef   = fermi_ha * _HA_TO_EV

        n_occ = int(n_elec // 2)   # number of occupied bands (spin-paired)
        if n_occ == 0 or n_occ >= eigs.shape[1]:
            return {'gap': 0.0, 'direct_gap': 0.0, 'vbm': ef, 'cbm': ef,
                    'is_metal': True, 'error': 'Band index out of range'}

        vbm_per_k = eigs[:, n_occ - 1]   # highest occupied band at each k
        cbm_per_k = eigs[:, n_occ]       # lowest unoccupied band at each k

        vbm = float(np.max(vbm_per_k))
        cbm = float(np.min(cbm_per_k))
        gap = cbm - vbm

        # Direct gap: min over k of (E_cbm,k - E_vbm,k)
        direct_gap = float(np.min(cbm_per_k - vbm_per_k))

        is_metal = gap <= 0.0
        return {
            'gap':        max(0.0, gap),
            'direct_gap': max(0.0, direct_gap),
            'vbm':        vbm - ef,   # relative to Fermi level
            'cbm':        cbm - ef,
            'is_metal':   is_metal,
        }

    # ------------------------------------------------------------------
    # Effective mass via parabolic fitting
    # ------------------------------------------------------------------

    def _effective_masses(self, kpoints_cart, energies_ev,
                          hob_idx: int, lub_idx: int) -> dict:
        """Fit parabolas near VBM (hob_idx) and CBM (lub_idx) to get m*.

        Uses the 7 k-points closest to each extremum along the band path.
        Returns effective masses in units of mₑ (electron rest mass).
        Returns None for each mass if fitting fails.
        """
        kpts  = np.array(kpoints_cart)  # (n_k, 3) Å⁻¹
        eigs  = np.array(energies_ev)   # (n_k, n_bands)

        def _fit_mass(band_idx: int, find_max: bool) -> Optional[float]:
            E = eigs[:, band_idx]
            k_idx = int(np.argmax(E) if find_max else np.argmin(E))
            k0 = kpts[k_idx]

            # Distances from extremum along path
            dists = np.linalg.norm(kpts - k0, axis=1)

            # Use up to 9 neighbouring k-points (excluding the extremum itself)
            # within a small radius
            order = np.argsort(dists)
            near = order[:min(9, len(order))]
            k_near = dists[near]
            E_near = E[near]

            if len(k_near) < 3:
                return None
            try:
                coeffs = np.polyfit(k_near, E_near, 2)  # E = a*k² + b*k + c
                a = coeffs[0]                            # eV·Å²
                if abs(a) < 1e-10:
                    return None
                m_eff = _HBAR2_2ME / abs(a)             # dimensionless (mₑ units)
                return float(m_eff) if 0.01 < m_eff < 100.0 else None
            except Exception:
                return None

        return {
            'm_electron': _fit_mass(lub_idx, find_max=False),
            'm_hole':     _fit_mass(hob_idx, find_max=True),
        }

    # ------------------------------------------------------------------
    # Formation energy
    # ------------------------------------------------------------------

    def _formation_energy(self, structure, total_energy_ev_per_atom) -> float:
        """Compute formation energy (eV/atom) vs PBE elemental references.

        ΔHf = E_total/N - Σ(xᵢ × μᵢ)
        where xᵢ = fraction of element i, μᵢ = reference energy per atom.
        """
        n = structure.num_sites
        elemental_sum = 0.0
        for site in structure:
            sym = site.specie.symbol
            ref = PBE_ELEMENTAL_REFERENCES.get(sym)
            if ref is None:
                logger.warning(f"No PBE reference for {sym}; using 0.0")
                ref = 0.0
            elemental_sum += ref
        return total_energy_ev_per_atom - elemental_sum / n

    # ------------------------------------------------------------------
    # pw.x input writer
    # ------------------------------------------------------------------

    def _write_pw_input(self, structure, pseudos, work, calculation,
                        kpoints_card, nbnd=None) -> Path:
        """Write a QE pw.x input file and return its path."""
        outdir = str(work / 'outdir')
        prefix = 'crystal'
        n_atoms = structure.num_sites
        species = sorted({s.specie.symbol for s in structure})
        n_types = len(species)

        # SCF uses smearing for convergence robustness.
        # NSCF uses tetrahedra (automatic k-mesh) for clean gap determination.
        # Bands uses fixed (explicit k-path) for effective-mass fitting.
        if calculation == 'scf':
            occ_block = {
                'occupations': "'smearing'",
                'smearing':    f"'{self.smearing}'",
                'degauss':     self.degauss,
            }
        elif calculation == 'nscf':
            occ_block = {
                'occupations': "'tetrahedra'",
            }
        else:  # bands
            occ_block = {
                'occupations': "'fixed'",
            }

        system_block = {
            'ibrav':   0,
            'nat':     n_atoms,
            'ntyp':    n_types,
            'ecutwfc': self.ecutwfc,
            'ecutrho': self.ecutrho,
            **occ_block,
        }
        if nbnd is not None:
            system_block['nbnd'] = nbnd

        lines = []

        # &CONTROL
        lines.append("&CONTROL")
        lines.append(f"  calculation = '{calculation}'")
        lines.append(f"  prefix      = '{prefix}'")
        lines.append(f"  outdir      = '{outdir}'")
        lines.append(f"  pseudo_dir  = '{self.pseudo_dir}'")
        lines.append(f"  tprnfor     = .true.")
        lines.append(f"  tstress     = .true.")
        lines.append(f"  verbosity   = 'low'")
        lines.append("/")

        # &SYSTEM
        lines.append("&SYSTEM")
        for k, v in system_block.items():
            lines.append(f"  {k:<12} = {v}")
        lines.append("/")

        # &ELECTRONS
        lines.append("&ELECTRONS")
        lines.append(f"  conv_thr    = {self.conv_thr:.2e}")
        lines.append(f"  mixing_mode = 'plain'")
        lines.append(f"  mixing_beta = 0.7")
        lines.append("/")

        # ATOMIC_SPECIES
        lines.append("ATOMIC_SPECIES")
        for sym in species:
            from pymatgen.core import Element
            mass = Element(sym).atomic_mass
            upf = pseudos[sym].name
            lines.append(f"  {sym:<4}  {float(mass):.4f}  {upf}")

        # CELL_PARAMETERS angstrom
        lines.append("CELL_PARAMETERS angstrom")
        for vec in structure.lattice.matrix:
            lines.append(f"  {vec[0]:.10f}  {vec[1]:.10f}  {vec[2]:.10f}")

        # ATOMIC_POSITIONS crystal
        lines.append("ATOMIC_POSITIONS crystal")
        for site in structure:
            fc = site.frac_coords
            lines.append(f"  {site.specie.symbol:<4} "
                         f"{fc[0]:.10f}  {fc[1]:.10f}  {fc[2]:.10f}")

        # K_POINTS (provided as a pre-formatted card)
        lines.append(kpoints_card.rstrip())

        content = '\n'.join(lines) + '\n'
        suffix = {'scf': 'scf', 'nscf': 'nscf', 'bands': 'bands'}.get(
            calculation, calculation)
        inp_path = work / f'pw.{suffix}.in'
        inp_path.write_text(content)
        return inp_path

    # ------------------------------------------------------------------
    # pw.x subprocess runner
    # ------------------------------------------------------------------

    def _run_pw_x(self, input_file: Path, output_file: Path,
                  work: Path) -> bool:
        """Run pw.x on *input_file*, write stdout+stderr to *output_file*.

        Respects QE_BINARY (path to pw.x) and QE_NPROC (MPI ranks) env vars.
        """
        pw_x = os.environ.get('QE_BINARY', '')
        if not pw_x:
            # Auto-detect: prefer full path in same conda env as this Python
            _candidate = Path(sys.executable).parent / 'pw.x'
            pw_x = str(_candidate) if _candidate.exists() else 'pw.x'
        nproc = int(os.environ.get('QE_NPROC', '1'))

        if nproc > 1:
            cmd = ['mpirun', '--oversubscribe', '-n', str(nproc), pw_x, '-in', str(input_file)]
        else:
            cmd = [pw_x, '-in', str(input_file)]

        try:
            with open(output_file, 'w') as fout:
                result = subprocess.run(
                    cmd,
                    stdout=fout,
                    stderr=subprocess.STDOUT,
                    cwd=str(work),
                    timeout=7200,
                )
            if self.verbose:
                print(output_file.read_text()[-3000:])
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            logger.error(f"pw.x timed out for {input_file.name}")
            return False
        except FileNotFoundError:
            logger.error(f"pw.x not found: {pw_x}. Install with: "
                         "conda install -n mol-evo -c conda-forge qe")
            return False

    # ------------------------------------------------------------------
    # XML output parser
    # ------------------------------------------------------------------

    def _parse_xml(self, work: Path) -> Optional[dict]:
        """Parse QE's data-file-schema.xml for energies and eigenvalues."""
        save_dir = work / 'outdir' / 'crystal.save'
        xml_file = save_dir / 'data-file-schema.xml'
        if not xml_file.exists():
            # Try alternate location
            for candidate in (work / 'outdir').glob('*.save/data-file-schema.xml'):
                xml_file = candidate
                break
        if not xml_file.exists():
            logger.warning(f"QE XML not found in {work / 'outdir'}")
            return None

        try:
            tree = ET.parse(xml_file)
        except ET.ParseError as e:
            logger.warning(f"XML parse error: {e}")
            return None

        root = tree.getroot()

        # Strip XML namespaces for easier searching
        def _strip_ns(element):
            for el in element.iter():
                el.tag = el.tag.split('}')[-1] if '}' in el.tag else el.tag
        _strip_ns(root)

        result = {}

        # Total energy (Hartree)
        etot_el = root.find('.//total_energy/etot')
        if etot_el is not None:
            result['total_energy_ha'] = float(etot_el.text)

        # Fermi energy (Hartree)
        ef_el = root.find('.//fermi_energy')
        if ef_el is not None:
            result['fermi_energy_ha'] = float(ef_el.text)
        else:
            result['fermi_energy_ha'] = 0.0

        # Eigenvalues: list of lists, one per k-point
        eigenvalues = []
        for ks_el in root.findall('.//ks_energies'):
            eig_el = ks_el.find('eigenvalues')
            if eig_el is not None and eig_el.text:
                vals = [float(x) for x in eig_el.text.split()]
                eigenvalues.append(vals)
        if eigenvalues:
            result['eigenvalues_ha'] = eigenvalues

        return result if result else None

    # ------------------------------------------------------------------
    # Pseudopotential resolver
    # ------------------------------------------------------------------

    def _resolve_pseudos(self, structure) -> Tuple[dict, dict, Optional[str]]:
        """Find pseudopotential files and read valence electrons for structure.

        Returns (pseudos_dict, valences_dict, error_string_or_None).
        """
        pseudos = {}
        valences = {}
        missing = []
        for sym in {s.specie.symbol for s in structure}:
            upf = find_pseudo(sym, self.pseudo_dir)
            if upf is None:
                missing.append(sym)
            else:
                pseudos[sym] = upf
                z = read_upf_valence(upf) or _DEFAULT_VALENCE.get(sym, 4)
                valences[sym] = z
        if missing:
            err = (f"Missing pseudopotentials for: {', '.join(missing)} "
                   f"in {self.pseudo_dir}. "
                   f"Run: download_sssp_pseudos({missing}, '{self.pseudo_dir}')")
            return {}, {}, err
        return pseudos, valences, None

    # ------------------------------------------------------------------
    # Structural properties (pymatgen, no QE)
    # ------------------------------------------------------------------

    @staticmethod
    def _structural_props(structure) -> dict:
        props = {
            'n_sites':   structure.num_sites,
            'n_species': len(structure.composition.elements),
            'volume':    float(structure.volume),
            'density':   float(structure.density),
            'spacegroup': 0,
        }
        try:
            from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
            props['spacegroup'] = int(
                SpacegroupAnalyzer(structure).get_space_group_number())
        except Exception:
            pass
        return props

    @staticmethod
    def _empty_props(structure) -> dict:
        return {
            'total_energy':      0.0,
            'formation_energy':  0.0,
            'bandgap':           0.0,
            'bandgap_direct':    0.0,
            'vbm':               0.0,
            'cbm':               0.0,
            'is_metal':          False,
            'effective_mass_e':  None,
            'effective_mass_h':  None,
            'band_structure':    None,
            # Elastic / hardness (None until calc_hardness=True)
            'bulk_modulus':               None,
            'shear_modulus':              None,
            'youngs_modulus':             None,
            'poisson_ratio':              None,
            'hardness_vickers':           None,
            'elastic_tensor':             None,
            # Derived elastic properties (None until calc_hardness=True)
            'pugh_ratio':                 None,
            'cauchy_pressure':            None,
            'sound_velocity_longitudinal': None,
            'sound_velocity_transverse':  None,
            'debye_temperature':          None,
            'fracture_toughness':         None,
            # Phonon stability (None until calc_phonon_stability=True)
            'is_dynamically_stable':      None,
            'n_imaginary_modes':          None,
            'phonon_frequencies_cm1':     None,
            # Tier 1: extra elastic-derived (None until calc_hardness=True)
            'thermal_conductivity_slack': None,
            'elastic_anisotropy_zener':   None,
            'peierls_stress_gpa':         None,
            # Tier 2: carrier physics (None if bandgap/masses unavailable)
            'gap_type':                   None,
            'carrier_lifetime_radiative': None,
            'recombination_coefficient_B': None,
            'intrinsic_carrier_density':  None,
            # Tier 3: deformation potential mobility (None until calc_mobility=True)
            'deformation_potential_e':    None,
            'deformation_potential_h':    None,
            'mobility_e':                 None,
            'mobility_h':                 None,
            # Tier 4: thermodynamic stability (None until calc_hull_distance=True)
            'e_above_hull':               None,
            'is_thermodynamically_stable': None,
            # Structural
            'n_sites':           structure.num_sites,
            'n_species':         len(structure.composition.elements),
            'volume':            float(structure.volume),
            'density':           float(structure.density),
            'spacegroup':        0,
            'error':             None,
        }

    @staticmethod
    def _tail_error(output_file: Path, n: int = 20) -> str:
        """Return the last *n* lines of an output file as an error string."""
        try:
            lines = output_file.read_text().splitlines()
            return '\n'.join(lines[-n:])
        except Exception:
            return f"Could not read {output_file}"

    # ------------------------------------------------------------------
    # Elastic constants and Vickers hardness
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_stress_kbar(output_file: Path) -> Optional[np.ndarray]:
        """Parse the final stress tensor (kbar) from a pw.x output file.

        QE prints after SCF convergence:
            total   stress  (Ry/bohr**3)          (kbar)   P= ...
              v11  v12  v13     s11  s12  s13
              v21  v22  v23     s21  s22  s23
              v31  v32  v33     s31  s32  s33

        Returns 3×3 ndarray (kbar) or None if not found.
        """
        try:
            lines = output_file.read_text().splitlines()
        except Exception:
            return None

        stress = None
        for i, line in enumerate(lines):
            if 'total   stress' in line and '(kbar)' in line:
                try:
                    rows = []
                    for j in range(1, 4):
                        parts = lines[i + j].split()
                        if len(parts) >= 6:
                            # Columns 3-5 (0-indexed) are the kbar values
                            rows.append([float(parts[3]),
                                         float(parts[4]),
                                         float(parts[5])])
                    if len(rows) == 3:
                        stress = np.array(rows)
                except (IndexError, ValueError):
                    pass
        return stress  # last occurrence = converged stress

    @staticmethod
    def _deform_structure(structure, voigt_idx: int, delta: float):
        """Return a new Structure with one Voigt strain mode applied.

        Args:
            voigt_idx: 0=ε₁₁, 1=ε₂₂, 2=ε₃₃, 3=γ₂₃, 4=γ₁₃, 5=γ₁₂
            delta:     Engineering strain amplitude (signed).

        Fractional atomic coordinates are kept fixed; only the lattice
        matrix is deformed (frozen-ion elastic constants).
        """
        from pymatgen.core import Structure as PMGStructure, Lattice as PMGLattice

        eps = np.zeros((3, 3))
        if voigt_idx == 0:
            eps[0, 0] = delta
        elif voigt_idx == 1:
            eps[1, 1] = delta
        elif voigt_idx == 2:
            eps[2, 2] = delta
        elif voigt_idx == 3:                   # engineering γ₂₃ = 2ε₂₃
            eps[1, 2] = eps[2, 1] = delta / 2
        elif voigt_idx == 4:                   # engineering γ₁₃ = 2ε₁₃
            eps[0, 2] = eps[2, 0] = delta / 2
        elif voigt_idx == 5:                   # engineering γ₁₂ = 2ε₁₂
            eps[0, 1] = eps[1, 0] = delta / 2

        # pymatgen rows = lattice vectors; new lattice A' = A @ F^T, F = I + ε
        F = np.eye(3) + eps
        new_matrix = structure.lattice.matrix @ F.T
        return PMGStructure(
            PMGLattice(new_matrix),
            [s.species_string for s in structure],
            structure.frac_coords,
            coords_are_cartesian=False,
        )

    def _compute_elastic_tensor(self, structure, pseudos: dict,
                                 valences: dict, work: Path
                                 ) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """Compute the 6×6 elastic tensor (GPa) via finite central differences.

        Applies each of the 6 Voigt strain modes at ±strain_delta, runs one
        SCF per deformation (12 total), extracts the stress tensor from each
        output, and computes Cᵢⱼ = Δσᵢ / Δεⱼ.

        Returns (C_GPa_6x6, None) on success, (None, error_string) on failure.
        """
        delta = self.strain_delta
        C = np.zeros((6, 6))

        for j in range(6):
            stresses: dict = {}
            for sign in (+1, -1):
                d = sign * delta
                deformed = self._deform_structure(structure, j, d)
                tag = f'strain_{j}_{"p" if sign > 0 else "m"}'
                s_work = work / tag
                s_work.mkdir(exist_ok=True)

                kcard = (f"K_POINTS automatic\n"
                         f"  {self.kpoints_scf[0]} {self.kpoints_scf[1]} "
                         f"{self.kpoints_scf[2]}  0 0 0\n")
                inp = self._write_pw_input(deformed, pseudos, s_work,
                                           calculation='scf',
                                           kpoints_card=kcard)
                out = s_work / 'pw.scf.out'
                ok = self._run_pw_x(inp, out, s_work)
                if not ok:
                    return None, f"Elastic SCF failed: {tag}"

                σ = self._parse_stress_kbar(out)
                if σ is None:
                    return None, f"Stress parse failed: {tag}"
                stresses[sign] = σ / 10.0   # kbar → GPa

            # Central difference: ΔC_j = (σ(+δ) - σ(-δ)) / (2δ)
            Δσ = stresses[+1] - stresses[-1]  # 3×3 GPa
            # Map 3×3 σ tensor → 6-component Voigt column
            Δσ_voigt = np.array([
                Δσ[0, 0],  # σ₁₁
                Δσ[1, 1],  # σ₂₂
                Δσ[2, 2],  # σ₃₃
                Δσ[1, 2],  # σ₂₃
                Δσ[0, 2],  # σ₁₃
                Δσ[0, 1],  # σ₁₂
            ])
            C[:, j] = Δσ_voigt / (2 * delta)

        # Enforce symmetry: C = (C + Cᵀ) / 2
        C = (C + C.T) / 2
        return C, None

    @staticmethod
    def _voigt_reuss_hill(C: np.ndarray) -> Tuple[float, float, float, float]:
        """Voigt-Reuss-Hill elastic averages from a 6×6 Voigt elastic tensor (GPa).

        Returns (B_GPa, G_GPa, E_GPa, nu).

        Voigt (upper) and Reuss (lower) bounds are averaged for B and G.
        Young's modulus E = 9BG/(3B+G) and Poisson's ratio ν = (3B-2G)/(2(3B+G))
        are derived from the Hill averages.
        """
        # Voigt averages (indices in 0-based Voigt notation)
        B_V = (C[0,0] + C[1,1] + C[2,2]
               + 2*(C[0,1] + C[1,2] + C[0,2])) / 9.0
        G_V = (C[0,0] + C[1,1] + C[2,2]
               - C[0,1] - C[1,2] - C[0,2]
               + 3*(C[3,3] + C[4,4] + C[5,5])) / 15.0

        # Reuss averages via compliance tensor S = C⁻¹
        try:
            S = np.linalg.inv(C)
            B_R_inv = (S[0,0] + S[1,1] + S[2,2]
                       + 2*(S[0,1] + S[1,2] + S[0,2]))
            G_R_inv = (4*(S[0,0] + S[1,1] + S[2,2])
                       - 4*(S[0,1] + S[1,2] + S[0,2])
                       + 3*(S[3,3] + S[4,4] + S[5,5])) / 15.0
            B_R = 1.0 / B_R_inv if abs(B_R_inv) > 1e-12 else B_V
            G_R = 1.0 / G_R_inv if abs(G_R_inv) > 1e-12 else G_V
        except np.linalg.LinAlgError:
            B_R, G_R = B_V, G_V

        # Hill (arithmetic mean)
        B = max((B_V + B_R) / 2.0, 0.0)
        G = max((G_V + G_R) / 2.0, 0.0)

        denom = 3.0 * B + G
        E  = 9.0 * B * G / denom if denom > 0 else 0.0
        nu = (3.0 * B - 2.0 * G) / (2.0 * denom) if denom > 0 else 0.0

        return B, G, E, nu

    @staticmethod
    def _chen_hardness(B: float, G: float) -> float:
        """Vickers hardness (GPa) via the Chen et al. (2011) model.

        H_v = 2(k²G)^0.585 − 3  [GPa],  k = G/B (Pugh's ratio)

        Reference: X.-Q. Chen et al., Intermetallics 19 (2011) 1275–1281.

        Benchmarks (PBE elastic constants):
          Diamond:  ~87 GPa   Si:  ~9 GPa
          SiC (3C): ~27 GPa   GaAs: ~7 GPa   AlN (WZ): ~18 GPa
        """
        if B <= 0.0 or G <= 0.0:
            return 0.0
        k = G / B
        return max(0.0, 2.0 * (k * k * G) ** 0.585 - 3.0)

    # ------------------------------------------------------------------
    # Derived elastic properties (no extra QE calculation)
    # ------------------------------------------------------------------

    @staticmethod
    def _debye_temperature(B: float, G: float, density: float,
                            n_sites: int, volume_ang3: float) -> float:
        """Debye temperature θ_D (K) from Voigt-Reuss-Hill averaged B and G.

        Uses the Debye model with mean sound velocity:
          v_m = [(1/v_l³ + 2/v_t³)/3]^{-1/3}
          v_l = √((B + 4G/3)/ρ)   [longitudinal]
          v_t = √(G/ρ)             [transverse]

        Args:
            B, G:        VRH moduli (GPa)
            density:     mass density (g/cm³)
            n_sites:     atoms per unit cell
            volume_ang3: unit cell volume (Å³)
        """
        if B <= 0.0 or G <= 0.0 or density <= 0.0 or volume_ang3 <= 0.0:
            return 0.0
        rho = density * 1e3                          # g/cm³ → kg/m³
        v_l = np.sqrt((B + 4.0 * G / 3.0) * 1e9 / rho)   # m/s
        v_t = np.sqrt(G * 1e9 / rho)
        if v_l <= 0.0 or v_t <= 0.0:
            return 0.0
        v_m_inv3 = (1.0 / v_l**3 + 2.0 / v_t**3) / 3.0
        if v_m_inv3 <= 0.0:
            return 0.0
        v_m    = v_m_inv3 ** (-1.0 / 3.0)
        V_m3   = volume_ang3 * 1e-30               # Å³ → m³
        n_dens = n_sites / V_m3                    # atoms m⁻³
        return float(_HBAR_KB * (6.0 * np.pi**2 * n_dens) ** (1.0 / 3.0) * v_m)

    @staticmethod
    def _derived_elastic_props(B: float, G: float, C: np.ndarray,
                                H_v: float, density: float,
                                n_sites: int, volume_ang3: float) -> dict:
        """Properties derived analytically from elastic constants.

        Computed without any additional QE run — only requires the already-
        computed VRH moduli, elastic tensor C, hardness H_v, and structural
        data.

        Returns dict with:
          pugh_ratio         — G/B; >0.57 brittle, <0.57 ductile (Pugh 1954)
          cauchy_pressure    — C₁₂−C₄₄ (GPa); >0 metallic bonding, <0 covalent
          sound_velocity_longitudinal — v_l (m/s)
          sound_velocity_transverse   — v_t (m/s)
          debye_temperature  — θ_D (K) from v_l and v_t
          fracture_toughness — K_Ic (MPa√m) Mazhnik-Oganov 2019: 0.374√(G·H_v)
        """
        props: dict = {}
        rho = density * 1e3   # kg/m³

        # Pugh ratio
        props['pugh_ratio'] = float(G / B) if B > 0.0 else 0.0

        # Cauchy pressure: C₁₂ − C₄₄  (Voigt indices [0,1] and [3,3])
        props['cauchy_pressure'] = float(C[0, 1] - C[3, 3])

        # Sound velocities (m/s)
        if rho > 0.0 and B > 0.0 and G > 0.0:
            v_l = np.sqrt((B + 4.0 * G / 3.0) * 1e9 / rho)
            v_t = np.sqrt(G * 1e9 / rho)
        else:
            v_l = v_t = 0.0
        props['sound_velocity_longitudinal'] = float(v_l)
        props['sound_velocity_transverse']   = float(v_t)

        # Debye temperature
        props['debye_temperature'] = QECrystalCalculator._debye_temperature(
            B, G, density, n_sites, volume_ang3)

        # Fracture toughness — Mazhnik & Oganov, J. Appl. Phys. 126 (2019) 125109
        # K_Ic = 0.374 × √(G [GPa] × H_v [GPa])  →  MPa√m
        if H_v is not None and H_v > 0.0 and G > 0.0:
            props['fracture_toughness'] = float(0.374 * np.sqrt(G * H_v))
        else:
            props['fracture_toughness'] = 0.0

        return props

    @staticmethod
    def _tier1_elastic_extras(B: float, G: float, C: np.ndarray,
                               H_v: float, nu: float, density: float,
                               n_sites: int, volume_ang3: float) -> dict:
        """Additional elastic-derived semiconductor properties (no extra QE runs).

        Computes thermal conductivity (Slack model), Zener elastic anisotropy,
        and Peierls-Nabarro dislocation stress from the already-computed
        VRH moduli, elastic tensor, and structural data.

        Returns dict with:
          thermal_conductivity_slack  — κ (W/m·K) Slack (1973) model
          elastic_anisotropy_zener    — A_Z = 2C₄₄/(C₁₁−C₁₂) (isotropic = 1)
          peierls_stress_gpa          — τ_PN (GPa) isotropic Peierls-Nabarro estimate
        """
        props: dict = {}

        # ------------------------------------------------------------------
        # Slack thermal conductivity:
        #   κ = A × M̄_a × θ_D³ × δ / (γ² × n^(2/3) × T)
        # Dugdale-MacDonald Grüneisen parameter:
        #   γ = 3(1+ν) / (2(2-3ν))
        # Prefactor A ≈ 3.1×10⁻⁶ calibrated for:
        #   M̄_a in amu, δ in Å, θ_D in K, T in K → κ in W/m·K
        # Ref: Slack (1973), Morelli & Slack (2006)
        # ------------------------------------------------------------------
        theta_D = QECrystalCalculator._debye_temperature(
            B, G, density, n_sites, volume_ang3)
        denom_nu = 2.0 - 3.0 * nu
        if theta_D and theta_D > 0.0 and abs(denom_nu) > 1e-9:
            gamma = 3.0 * (1.0 + nu) / (2.0 * denom_nu)
            # Mean atomic mass (amu): m̄ = ρ [g/cm³] × V [Å³] × 1e-24 / n_sites / 1.66054e-24
            M_a = density * (volume_ang3 * 1e-24) / n_sites / 1.66054e-24
            # Mean interatomic distance δ (Å) = (V/n)^(1/3)
            delta_ang = (volume_ang3 / n_sites) ** (1.0 / 3.0)
            T = 300.0  # room temperature (K)
            kappa = (3.1e-6 * M_a * theta_D**3 * delta_ang) / (
                gamma**2 * n_sites**(2.0 / 3.0) * T)
            props['thermal_conductivity_slack'] = float(max(kappa, 0.0))
        else:
            props['thermal_conductivity_slack'] = 0.0

        # ------------------------------------------------------------------
        # Zener elastic anisotropy: A_Z = 2C₄₄ / (C₁₁ − C₁₂)
        # A_Z = 1 → isotropic; >1 or <1 → anisotropic
        # ------------------------------------------------------------------
        c11_c12 = float(C[0, 0] - C[0, 1])
        if abs(c11_c12) > 1e-6:
            props['elastic_anisotropy_zener'] = float(2.0 * C[3, 3] / c11_c12)
        else:
            props['elastic_anisotropy_zener'] = None

        # ------------------------------------------------------------------
        # Peierls-Nabarro stress (isotropic estimate):
        #   τ_PN ≈ 2G × exp(−2π² / (1−ν))
        # With b = d = (V/n)^(1/3), the ratio d/b = 1 so the exponent
        # simplifies to −2π²/(1−ν).
        # Ref: Peierls (1940), Nabarro (1947)
        # ------------------------------------------------------------------
        if G > 0.0 and nu < 1.0:
            exponent = -2.0 * np.pi**2 / (1.0 - nu)
            tau_gpa = 2.0 * G * np.exp(exponent)
            props['peierls_stress_gpa'] = float(tau_gpa)
        else:
            props['peierls_stress_gpa'] = 0.0

        return props

    @staticmethod
    def _classify_gap_type(bandgap: float, bandgap_direct: float) -> str:
        """Classify gap type as 'metal', 'direct', or 'indirect'.

        Threshold: direct if the direct gap is within 50 meV of the indirect
        fundamental gap (i.e. the band extrema are at the same k-point).
        """
        if bandgap <= 0.01:
            return 'metal'
        if bandgap_direct <= 0.0:
            return 'indirect'
        return 'direct' if (bandgap_direct - bandgap) < 0.05 else 'indirect'

    @staticmethod
    def _radiative_lifetime(bandgap_ev: float,
                             me_star: Optional[float],
                             mh_star: Optional[float],
                             T: float = 300.0) -> Tuple[Optional[float],
                                                        Optional[float],
                                                        Optional[float]]:
        """Van Roosbroeck-Shockley simplified radiative carrier lifetime.

        Returns (tau_radiative [s], B_coefficient [cm³/s], n_i [cm⁻³]).

        Simplified expressions valid for direct-gap III-V / II-VI semiconductors:
          B  ≈ 1.8×10⁻⁸ / (n_r² × E_g [eV] × (m_e*·m_h*)^0.75 × T^1.5)   cm³/s
          n_i ≈ 4.9×10¹⁵ × (m_e*·m_h*)^0.75 × T^1.5 × exp(−E_g / 2k_BT)   cm⁻³
          τ   = 1 / (B × n_i)                                                 s

        These are order-of-magnitude estimates.  For quantitative accuracy use
        full VRS integration with the actual dielectric function.

        Ref: van Roosbroeck & Shockley, Phys. Rev. 94 (1954) 1558.
        """
        import math
        if bandgap_ev <= 0.0 or me_star is None or mh_star is None:
            return None, None, None
        if me_star <= 0.0 or mh_star <= 0.0:
            return None, None, None
        try:
            kT = _KB_EV * T
            mr = (me_star * mh_star) ** 0.75
            n_r = 3.5   # approximate refractive index (typical semiconductor)
            B_coeff = 1.8e-8 / (n_r**2 * bandgap_ev * mr * T**1.5)      # cm³/s
            ni = 4.9e15 * mr * T**1.5 * math.exp(-bandgap_ev / (2.0 * kT))  # cm⁻³
            tau = 1.0 / (B_coeff * ni) if (B_coeff > 0.0 and ni > 0.0) else None
            return tau, B_coeff, ni
        except (OverflowError, ZeroDivisionError, ValueError):
            return None, None, None

    def _compute_deformation_potential(self, structure, pseudos: dict,
                                        valences: dict,
                                        work: Path) -> Tuple[bool, dict]:
        """Compute electron/hole deformation potentials via finite-difference strain.

        Runs 4 QE calculations: SCF+NSCF at ε₁₁ = +0.5% and −0.5%.
        The deformation potential is:
          E₁ = (ε_edge(+δ) − ε_edge(−δ)) / (2δ)    [eV]
        where ε_edge is the CBM (electrons) or VBM (holes) energy and δ = 0.005.

        Returns (True, {'dp_e': eV, 'dp_h': eV}) on success,
                (False, {'error': str}) on failure.
        """
        delta = 0.005   # 0.5% uniaxial strain along a (ε₁₁)
        n_elec = sum(valences[s.specie.symbol] for s in structure)
        results: dict = {}
        for sign, label in ((+1, 'pos'), (-1, 'neg')):
            strained = self._deform_structure(structure, voigt_idx=0,
                                               delta=float(sign) * delta)
            work_dp = work / f'dp_{label}'
            work_dp.mkdir(exist_ok=True)
            scf_ok, scf_data = self._run_scf(strained, pseudos, work_dp)
            if not scf_ok:
                return False, {'error': f'DP SCF {label} failed: '
                                        f'{scf_data.get("error", "unknown")}'}
            nscf_ok, nscf_data = self._run_nscf(
                strained, pseudos, work_dp, n_elec)
            if not nscf_ok:
                return False, {'error': f'DP NSCF {label} failed: '
                                        f'{nscf_data.get("error", "unknown")}'}
            results[label] = nscf_data

        dp_e = (results['pos']['cbm'] - results['neg']['cbm']) / (2.0 * delta)
        dp_h = (results['pos']['vbm'] - results['neg']['vbm']) / (2.0 * delta)
        return True, {'dp_e': abs(float(dp_e)), 'dp_h': abs(float(dp_h))}

    @staticmethod
    def _dp_mobility(me_star: Optional[float],
                     mh_star: Optional[float],
                     dp_e: float,
                     dp_h: float,
                     C_ii_gpa: float,
                     T: float = 300.0) -> Tuple[Optional[float], Optional[float]]:
        """Bardeen-Shockley deformation potential mobility (cm²/V·s).

        Formula (3D isotropic, parabolic band):
          μ = (2√(2π) × e × ℏ⁴ × C_ii) / (3 × (m*)^(5/2) × E₁² × (k_BT)^(3/2))

        Args:
          me_star / mh_star : effective masses in units of m₀ (electron mass)
          dp_e / dp_h       : deformation potentials (eV)
          C_ii_gpa          : elastic stiffness C₁₁ (GPa)
          T                 : temperature (K), default 300

        Returns (mu_e, mu_h) in cm²/V·s, or (None, None) on failure.

        Ref: Bardeen & Shockley, Phys. Rev. 80 (1950) 72.
        """
        import math
        if None in (me_star, mh_star) or C_ii_gpa <= 0.0:
            return None, None
        if me_star <= 0.0 or mh_star <= 0.0 or dp_e <= 0.0 or dp_h <= 0.0:
            return None, None
        try:
            C_ii = C_ii_gpa * 1.0e9             # Pa
            kT_J = 1.380649e-23 * T              # J
            prefactor = (2.0 * math.sqrt(2.0 * math.pi) *
                         _E_C * _HBAR_J**4 * C_ii / 3.0)
            mu_e_si = prefactor / (
                (me_star * _M0_KG)**2.5 * (dp_e * _E_C)**2 * kT_J**1.5)
            mu_h_si = prefactor / (
                (mh_star * _M0_KG)**2.5 * (dp_h * _E_C)**2 * kT_J**1.5)
            return float(mu_e_si) * 1.0e4, float(mu_h_si) * 1.0e4  # m²/V·s → cm²/V·s
        except (OverflowError, ZeroDivisionError, ValueError):
            return None, None

    def _compute_hull_distance(self, structure) -> Tuple[Optional[float], Optional[bool]]:
        """Compute e_above_hull via CHGNet + pymatgen PhaseDiagram.

        Surveys competing prototype structures for the same chemical system,
        predicts their energies with CHGNet, builds a local PhaseDiagram,
        and returns the hull distance for *structure*.

        The competing-entry survey is cached per frozenset of element symbols
        to avoid repeated CHGNet calls for the same chemistry.

        Returns (e_above_hull [eV/atom], is_thermodynamically_stable) or
                (None, None) on failure.
        """
        elements = frozenset(s.specie.symbol for s in structure)

        # Lazy-load CHGNet model
        if self._chgnet_model is None:
            try:
                from chgnet.model import CHGNet
                self._chgnet_model = CHGNet.load()
            except ImportError:
                logger.warning("CHGNet not installed; cannot compute hull distance. "
                                "Install with: pip install chgnet")
                return None, None
            except Exception as exc:
                logger.warning(f"CHGNet load failed: {exc}")
                return None, None

        # Build competing entry list (cached per chemical system)
        if elements not in self._hull_cache:
            self._hull_cache[elements] = self._generate_competing_entries(
                elements, self._chgnet_model)

        competing_entries = self._hull_cache[elements]
        if not competing_entries:
            return None, None

        try:
            from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry

            # Predict formation energy of query structure.
            # CHGNet predict_structure returns pred['e'] in eV/atom.
            pred = self._chgnet_model.predict_structure(structure)
            e_per_atom = float(pred['e'])           # eV/atom (CHGNet output)
            query_entry = PDEntry(
                structure.composition,
                e_per_atom * structure.num_sites)   # total eV for this composition

            all_entries = list(competing_entries) + [query_entry]
            pd = PhaseDiagram(all_entries)
            e_hull = float(pd.get_e_above_hull(query_entry))
            return e_hull, e_hull < 0.05   # stable if within 50 meV/atom of hull

        except Exception as exc:
            logger.warning(f"Hull distance calculation failed: {exc}")
            return None, None

    @staticmethod
    def _generate_competing_entries(elements, chgnet_model) -> list:
        """Build a minimal competing-phase entry set for *elements*.

        Creates prototype crystal structures for each element and binary
        compound, predicts formation energies with CHGNet, and returns
        a list of pymatgen PDEntry objects for PhaseDiagram construction.
        """
        import itertools
        from pymatgen.core import Structure, Lattice, Composition
        from pymatgen.analysis.phase_diagram import PDEntry

        # Prototype structures for common semiconductor/oxide elements
        _SIMPLE_STRUCTS = {
            'Si': Structure(Lattice.cubic(5.43), ['Si', 'Si'],
                            [[0, 0, 0], [0.25, 0.25, 0.25]]),
            'Ge': Structure(Lattice.cubic(5.66), ['Ge', 'Ge'],
                            [[0, 0, 0], [0.25, 0.25, 0.25]]),
            'C':  Structure(Lattice.cubic(3.57), ['C', 'C'],
                            [[0, 0, 0], [0.25, 0.25, 0.25]]),
            'Ga': Structure(Lattice.cubic(4.05), ['Ga'], [[0, 0, 0]]),
            'In': Structure(Lattice.cubic(4.58), ['In'], [[0, 0, 0]]),
            'As': Structure(Lattice.cubic(5.65), ['As'], [[0, 0, 0]]),
            'Al': Structure(Lattice.cubic(4.05), ['Al'], [[0, 0, 0]]),
            'Sb': Structure(Lattice.cubic(6.09), ['Sb'], [[0, 0, 0]]),
            'N':  Structure(Lattice.cubic(5.0),  ['N', 'N'],
                            [[0, 0, 0], [0.5, 0.5, 0.5]]),
            'P':  Structure(Lattice.orthorhombic(3.31, 4.38, 10.50),
                            ['P'] * 4,
                            [[0, 0, 0], [0.5, 0, 0], [0, 0.5, 0], [0.5, 0.5, 0]]),
            'Cd': Structure(Lattice.hexagonal(2.98, 5.62), ['Cd'], [[0, 0, 0]]),
            'Zn': Structure(Lattice.hexagonal(2.66, 4.95), ['Zn'], [[0, 0, 0]]),
            'Se': Structure(Lattice.hexagonal(4.37, 5.0),
                            ['Se', 'Se', 'Se'],
                            [[0.23, 0, 0.33], [0.77, 0, 0.67], [0, 0, 0]]),
            'Te': Structure(Lattice.hexagonal(4.46, 5.92),
                            ['Te', 'Te', 'Te'],
                            [[0.27, 0, 0.33], [0.73, 0, 0.67], [0, 0, 0]]),
            'Hg': Structure(Lattice.rhombohedral(3.48, 70.0), ['Hg'], [[0, 0, 0]]),
            'S':  Structure(Lattice.orthorhombic(10.4, 12.9, 24.5),
                            ['S'] * 8,
                            [[0.125, 0.125, 0.125], [0.375, 0.125, 0.125],
                             [0.625, 0.125, 0.125], [0.875, 0.125, 0.125],
                             [0.125, 0.625, 0.125], [0.375, 0.625, 0.125],
                             [0.625, 0.625, 0.125], [0.875, 0.625, 0.125]]),
        }

        entries: list = []
        elem_list = sorted(elements)

        # CHGNet predict_structure returns pred['e'] in eV/atom.
        # PDEntry takes total energy for the given composition.

        # Elemental references
        for el in elem_list:
            struct = _SIMPLE_STRUCTS.get(
                el, Structure(Lattice.cubic(4.0), [el], [[0, 0, 0]]))
            try:
                pred = chgnet_model.predict_structure(struct)
                e_per_atom = float(pred['e'])               # eV/atom
                entries.append(PDEntry(struct.composition,
                                       e_per_atom * struct.num_sites))
            except Exception:
                pass

        # Binary prototype compounds (CsCl + zinc-blende)
        for el_a, el_b in itertools.combinations(elem_list, 2):
            protos = [
                # CsCl (B2) prototype
                Structure(Lattice.cubic(3.5), [el_a, el_b],
                          [[0, 0, 0], [0.5, 0.5, 0.5]]),
                # Zinc-blende (B3) prototype
                Structure(Lattice.cubic(5.5), [el_a, el_b, el_a, el_b],
                          [[0, 0, 0], [0.25, 0.25, 0.25],
                           [0.5, 0.5, 0], [0.75, 0.75, 0.25]]),
            ]
            for p in protos:
                try:
                    pred = chgnet_model.predict_structure(p)
                    e_per_atom = float(pred['e'])           # eV/atom
                    entries.append(PDEntry(p.composition,
                                           e_per_atom * p.num_sites))
                except Exception:
                    pass

        return entries

    # ------------------------------------------------------------------
    # Phonon stability at Γ (ph.x)
    # ------------------------------------------------------------------

    def _run_phonon_gamma(self, structure, pseudos: dict,
                           work: Path) -> Tuple[bool, dict]:
        """Run ph.x at q=(0,0,0) to check for imaginary phonon modes.

        A structure with imaginary modes (ω² < 0, reported by QE as negative
        frequencies) is dynamically unstable — the equilibrium geometry is a
        saddle point of the energy surface.

        Returns (True, data_dict) on success, (False, {'error': ...}) on failure.
        data_dict keys: is_stable (bool), n_imaginary (int),
                        frequencies_cm1 (list[float]).
        """
        ph_inp = work / 'ph.gamma.in'
        ph_out = work / 'ph.gamma.out'
        outdir = str(work / 'outdir')

        content = (
            "Phonon at Gamma\n"
            "&INPUTPH\n"
            f"  tr2_ph  = 1.0d-8,\n"
            f"  epsil   = .false.,\n"
            f"  ldisp   = .false.,\n"
            f"  prefix  = 'crystal',\n"
            f"  outdir  = '{outdir}',\n"
            f"  fildyn  = 'crystal.dyn',\n"
            "/\n"
            "0.000000 0.000000 0.000000\n"
        )
        ph_inp.write_text(content)

        ok = self._run_ph_x(ph_inp, ph_out, work)
        if not ok:
            return False, {'error': self._tail_error(ph_out)}

        freqs = self._parse_phonon_frequencies(ph_out)
        if freqs is None or len(freqs) == 0:
            return False, {'error': 'Could not parse phonon frequencies from ph.x output'}

        # Acoustic modes at Γ are exactly zero for a perfect crystal; allow ±10 cm⁻¹
        # numerical noise. Modes beyond the 3 acoustic branches with ω < -10 cm⁻¹
        # are truly imaginary (the structure is unstable).
        _IMAGINARY_THRESH = -10.0   # cm⁻¹
        _ACOUSTIC_THRESH  = -50.0   # cm⁻¹ — strongly imaginary even for acoustic branch

        n_imaginary = sum(1 for f in freqs[3:] if f < _IMAGINARY_THRESH)
        n_imaginary += sum(1 for f in freqs[:3] if f < _ACOUSTIC_THRESH)

        return True, {
            'is_stable':       n_imaginary == 0,
            'n_imaginary':     n_imaginary,
            'frequencies_cm1': freqs,
        }

    def _run_ph_x(self, input_file: Path, output_file: Path,
                  work: Path) -> bool:
        """Run ph.x for a phonon calculation.

        Respects QE_PH_BINARY (path to ph.x) and QE_NPROC env vars.
        Falls back to searching for ph.x in the same conda env as pw.x.
        """
        ph_x = os.environ.get('QE_PH_BINARY', '')
        if not ph_x:
            candidate = Path(sys.executable).parent / 'ph.x'
            ph_x = str(candidate) if candidate.exists() else 'ph.x'
        nproc = int(os.environ.get('QE_NPROC', '1'))

        if nproc > 1:
            cmd = ['mpirun', '--oversubscribe', '-n', str(nproc), ph_x, '-in', str(input_file)]
        else:
            cmd = [ph_x, '-in', str(input_file)]

        try:
            with open(output_file, 'w') as fout:
                result = subprocess.run(
                    cmd,
                    stdout=fout,
                    stderr=subprocess.STDOUT,
                    cwd=str(work),
                    timeout=7200,
                )
            if self.verbose:
                print(output_file.read_text()[-2000:])
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            logger.error("ph.x timed out")
            return False
        except FileNotFoundError:
            logger.error(f"ph.x not found: {ph_x}. Install with: "
                         "conda install -n mol-evo -c conda-forge qe")
            return False

    @staticmethod
    def _parse_phonon_frequencies(output_file: Path) -> Optional[List[float]]:
        """Parse Γ-point phonon frequencies (cm⁻¹) from ph.x stdout.

        Matches lines like:
            freq (   1) =      -2.234 [THz] =     -74.497 [cm-1]

        Imaginary modes are reported by QE as negative values.
        Returns list of floats (cm⁻¹) or None if no matches found.
        """
        try:
            text = output_file.read_text()
        except Exception:
            return None
        pattern = re.compile(
            r'freq\s*\(\s*\d+\)\s*=\s*[-\d.]+\s*\[THz\]\s*=\s*([-\d.]+)\s*\[cm-1\]'
        )
        matches = pattern.findall(text)
        return [float(m) for m in matches] if matches else None


# ---------------------------------------------------------------------------
# Convenience: compute a single elemental reference energy
# ---------------------------------------------------------------------------

def calc_elemental_reference(element: str, pseudo_dir: Optional[str] = None,
                              ecutwfc: float = 80.0, **kwargs) -> float:
    """Compute the PBE energy/atom for the most stable phase of *element*.

    This should be run once per pseudopotential library to calibrate the
    formation energy reference.  Results should replace the corresponding
    entry in PBE_ELEMENTAL_REFERENCES.

    Returns energy in eV/atom (negative).
    """
    from pymatgen.core import Structure, Lattice

    # Simple elemental structures for reference calculation
    elemental_structures = {
        'Si': Structure(Lattice.cubic(5.43), ['Si', 'Si'],
                        [[0, 0, 0], [0.25, 0.25, 0.25]]),
        'Ge': Structure(Lattice.cubic(5.66), ['Ge', 'Ge'],
                        [[0, 0, 0], [0.25, 0.25, 0.25]]),
        'C':  Structure(Lattice.cubic(3.57), ['C', 'C'],
                        [[0, 0, 0], [0.25, 0.25, 0.25]]),
        'GaN_Ga': Structure(Lattice.cubic(4.00), ['Ga'], [[0, 0, 0]]),  # FCC Ga
        'GaN_N':  Structure(Lattice.hexagonal(2.46, 6.70), ['N', 'N'],
                            [[0, 0, 0.25], [0, 0, 0.75]]),              # N₂ in box
    }
    struct = elemental_structures.get(element)
    if struct is None:
        raise NotImplementedError(
            f"No prototype structure for element '{element}'. "
            "Add one to calc_elemental_reference().")

    calc = QECrystalCalculator(
        ecutwfc=ecutwfc,
        pseudo_dir=pseudo_dir,
        kpoints_scf=(8, 8, 8),
        **kwargs,
    )
    result = calc._run_scf(struct, *calc._resolve_pseudos(struct)[:2],
                           Path(tempfile.mkdtemp(prefix='qe_ref_')))
    if result[0]:
        return result[1]['total_energy_ev_per_atom']
    raise RuntimeError(f"SCF failed for elemental {element}: {result[1]}")
