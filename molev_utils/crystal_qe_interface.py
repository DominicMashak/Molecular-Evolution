"""
CrystalQEInterface — Quantum ESPRESSO evaluation for SLICES crystal genotypes.

Drop-in replacement for CrystalEvaluationInterface that uses DFT (QE pw.x)
instead of the CHGNet ML surrogate.  Returns an extended property dict that
includes:

  formation_energy    eV/atom  (DFT PBE, vs elemental refs)
  bandgap             eV       indirect fundamental gap (0 = metal)
  bandgap_direct      eV       smallest direct gap at any k-point
  vbm / cbm           eV       relative to Fermi level
  is_metal            bool
  effective_mass_e    mₑ       electron effective mass at CBM
  effective_mass_h    mₑ       hole effective mass at VBM (positive)
  band_structure      dict | None   full E(k) data for post-processing

The dict is compatible with CrystalEvaluationInterface output — all keys
present in that interface are present here too (num_atoms, num_bonds, etc.),
so algorithm archives work without changes.

Usage
-----
    # Setup once per run
    from molev_utils.crystal_qe_interface import CrystalQEInterface
    ci = CrystalQEInterface(
        pseudo_dir='/path/to/pseudos',    # required
        ecutwfc=60,
        kpoints_scf=(4, 4, 4),
        kpoints_nscf=(8, 8, 8),
    )

    # In the evaluation loop
    props = ci.calculate(slices_str, structure=structure)
    print(props['bandgap'], props['effective_mass_e'])

Pseudopotentials
----------------
Download SSSP efficiency pseudopotentials before the first run:

    from quantum_chemistry.calculators.qe_crystal import download_sssp_pseudos
    download_sssp_pseudos(['Si', 'Ge', 'Ga', 'As', 'N'], '/path/to/pseudos')

Or set the ESPRESSO_PSEUDO environment variable.
"""

from __future__ import annotations

import logging
import sys
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Resolve imports regardless of working directory
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent
_CALC_DIR = _HERE.parent / 'quantum_chemistry' / 'calculators'
if str(_CALC_DIR) not in sys.path:
    sys.path.insert(0, str(_CALC_DIR))
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


def _qe_error_props(slices_str: str, error: str) -> dict:
    """Return a zeroed props dict (archive-compatible) with an error message."""
    return {
        'slices':             slices_str,
        'smiles':             None,
        'num_atoms':          0,
        'num_bonds':          0,
        'n_sites':            0,
        'n_species':          0,
        'volume':             0.0,
        'density':            0.0,
        'spacegroup':         0,
        'total_energy':       0.0,
        'formation_energy':   0.0,
        'bandgap':            0.0,
        'bandgap_direct':     0.0,
        'vbm':                0.0,
        'cbm':                0.0,
        'is_metal':           False,
        'effective_mass_e':   None,
        'effective_mass_h':   None,
        'band_structure':     None,
        # Elastic / hardness
        'bulk_modulus':                None,
        'shear_modulus':               None,
        'youngs_modulus':              None,
        'poisson_ratio':               None,
        'hardness_vickers':            None,
        'elastic_tensor':              None,
        # Derived elastic
        'pugh_ratio':                  None,
        'cauchy_pressure':             None,
        'sound_velocity_longitudinal': None,
        'sound_velocity_transverse':   None,
        'debye_temperature':           None,
        'fracture_toughness':          None,
        # Phonon stability
        'is_dynamically_stable':       None,
        'n_imaginary_modes':           None,
        'phonon_frequencies_cm1':      None,
        # Tier 1: extra elastic-derived
        'thermal_conductivity_slack':  None,
        'elastic_anisotropy_zener':    None,
        'peierls_stress_gpa':          None,
        # Tier 2: carrier physics
        'gap_type':                    None,
        'carrier_lifetime_radiative':  None,
        'recombination_coefficient_B': None,
        'intrinsic_carrier_density':   None,
        # Tier 3: deformation potential mobility
        'deformation_potential_e':     None,
        'deformation_potential_h':     None,
        'mobility_e':                  None,
        'mobility_h':                  None,
        # Tier 4: thermodynamic stability
        'e_above_hull':                None,
        'is_thermodynamically_stable': None,
        'error':                       error,
    }


class CrystalQEInterface:
    """Evaluate a SLICES crystal genotype with Quantum ESPRESSO DFT.

    Args:
        pseudo_dir:     Path to UPF pseudopotential directory.  If None,
                        falls back to ESPRESSO_PSEUDO env var or
                        ~/.local/share/espresso/pseudo/.
        functional:     XC functional ('PBE' default).
        ecutwfc:        Plane-wave kinetic energy cutoff in Ry (default 60).
        ecutrho:        Charge density cutoff in Ry (default 480 = 8×ecutwfc).
        kpoints_scf:    Monkhorst-Pack k-grid for SCF (default (4,4,4)).
        kpoints_nscf:   Denser k-grid for bandgap (default (8,8,8)).
        nk_bands:       Approximate k-points along band path (default 100).
        work_dir:       Base directory for temporary QE work dirs.
        keep_workdir:   Preserve work directories for debugging.
        conv_thr:       SCF convergence threshold in Ry (default 1e-8).
        verbose:        Print QE output to stdout.
    """

    def __init__(
        self,
        pseudo_dir: Optional[str] = None,
        functional: str = 'PBE',
        ecutwfc: float = 60.0,
        ecutrho: float = 480.0,
        kpoints_scf: tuple = (4, 4, 4),
        kpoints_nscf: tuple = (8, 8, 8),
        nk_bands: int = 100,
        work_dir: Optional[str] = None,
        keep_workdir: bool = False,
        conv_thr: float = 1e-8,
        verbose: bool = False,
        calc_hardness: bool = False,
        strain_delta: float = 0.01,
        calc_phonon_stability: bool = False,
        calc_mobility: bool = False,
        calc_hull_distance: bool = False,
    ) -> None:
        self.verbose = verbose
        self._calc = None          # lazy-loaded QECrystalCalculator
        self._calc_kwargs = dict(
            pseudo_dir=pseudo_dir,
            functional=functional,
            ecutwfc=ecutwfc,
            ecutrho=ecutrho,
            kpoints_scf=kpoints_scf,
            kpoints_nscf=kpoints_nscf,
            nk_bands=nk_bands,
            work_dir=work_dir,
            keep_workdir=keep_workdir,
            conv_thr=conv_thr,
            verbose=verbose,
            calc_hardness=calc_hardness,
            strain_delta=strain_delta,
            calc_phonon_stability=calc_phonon_stability,
            calc_mobility=calc_mobility,
            calc_hull_distance=calc_hull_distance,
        )

    # ------------------------------------------------------------------
    # Lazy calculator loader
    # ------------------------------------------------------------------

    def _load_calculator(self) -> None:
        if self._calc is not None:
            return
        try:
            from qe_crystal import QECrystalCalculator
        except ImportError:
            try:
                from quantum_chemistry.calculators.qe_crystal import QECrystalCalculator
            except ImportError as exc:
                raise ImportError(
                    "QECrystalCalculator not found.  Ensure "
                    "quantum_chemistry/calculators/qe_crystal.py exists."
                ) from exc
        self._calc = QECrystalCalculator(**self._calc_kwargs)
        if self.verbose:
            logger.info("CrystalQEInterface: QECrystalCalculator initialised "
                        f"(ecutwfc={self._calc_kwargs['ecutwfc']} Ry, "
                        f"kSCF={self._calc_kwargs['kpoints_scf']}, "
                        f"kNSCF={self._calc_kwargs['kpoints_nscf']})")

    # ------------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------------

    def calculate(self, slices_str: str, structure=None) -> dict:
        """Evaluate a crystal structure with QE DFT.

        Args:
            slices_str: SLICES genotype string (used as the dict key 'slices').
            structure:  Pre-decoded pymatgen Structure.  If None, it is
                        reconstructed from *slices_str* via SLICESMutator.

        Returns:
            dict compatible with CrystalEvaluationInterface.calculate(),
            plus QE-specific keys: bandgap_direct, vbm, cbm, is_metal,
            effective_mass_e, effective_mass_h, band_structure, total_energy.
        """
        # ---- 1. Decode structure if not provided ---------------------------
        if structure is None:
            try:
                from slices_ops import SLICESMutator
            except ImportError:
                from molev_utils.slices_ops import SLICESMutator
            structure = SLICESMutator().to_structure(slices_str)

        if structure is None:
            return _qe_error_props(slices_str, 'Structure reconstruction failed')

        # ---- 2. Run QE workflow --------------------------------------------
        try:
            self._load_calculator()
        except ImportError as exc:
            return _qe_error_props(slices_str, str(exc))

        try:
            qe_props = self._calc.calculate(structure)
        except Exception as exc:
            logger.error(f"QE calculation failed: {exc}")
            return _qe_error_props(slices_str, f'QE error: {exc}')

        # ---- 3. Build output dict (archive-compatible) --------------------
        n_sites   = qe_props.get('n_sites',   structure.num_sites)
        n_species = qe_props.get('n_species', len(structure.composition.elements))

        return {
            # Archive compatibility keys (matches CrystalEvaluationInterface)
            'slices':            slices_str,
            'smiles':            None,
            'num_atoms':         n_sites,    # mapped for archive binning
            'num_bonds':         0,
            # Structural
            'n_sites':           n_sites,
            'n_species':         n_species,
            'volume':            qe_props.get('volume',   0.0),
            'density':           qe_props.get('density',  0.0),
            'spacegroup':        qe_props.get('spacegroup', 0),
            # Energetics
            'total_energy':      qe_props.get('total_energy',     0.0),
            'formation_energy':  qe_props.get('formation_energy', 0.0),
            # Electronic (QE-specific)
            'bandgap':           qe_props.get('bandgap',          0.0),
            'bandgap_direct':    qe_props.get('bandgap_direct',   0.0),
            'vbm':               qe_props.get('vbm',              0.0),
            'cbm':               qe_props.get('cbm',              0.0),
            'is_metal':          qe_props.get('is_metal',         False),
            'effective_mass_e':  qe_props.get('effective_mass_e', None),
            'effective_mass_h':  qe_props.get('effective_mass_h', None),
            'band_structure':    qe_props.get('band_structure',   None),
            # Elastic / hardness (populated when calc_hardness=True)
            'bulk_modulus':      qe_props.get('bulk_modulus',      None),
            'shear_modulus':     qe_props.get('shear_modulus',     None),
            'youngs_modulus':    qe_props.get('youngs_modulus',    None),
            'poisson_ratio':     qe_props.get('poisson_ratio',     None),
            'hardness_vickers':  qe_props.get('hardness_vickers',  None),
            'elastic_tensor':    qe_props.get('elastic_tensor',    None),
            # Derived elastic (populated when calc_hardness=True)
            'pugh_ratio':                  qe_props.get('pugh_ratio',                  None),
            'cauchy_pressure':             qe_props.get('cauchy_pressure',             None),
            'sound_velocity_longitudinal': qe_props.get('sound_velocity_longitudinal', None),
            'sound_velocity_transverse':   qe_props.get('sound_velocity_transverse',   None),
            'debye_temperature':           qe_props.get('debye_temperature',           None),
            'fracture_toughness':          qe_props.get('fracture_toughness',          None),
            # Phonon stability (populated when calc_phonon_stability=True)
            'is_dynamically_stable':       qe_props.get('is_dynamically_stable',       None),
            'n_imaginary_modes':           qe_props.get('n_imaginary_modes',           None),
            'phonon_frequencies_cm1':      qe_props.get('phonon_frequencies_cm1',      None),
            # Tier 1: extra elastic-derived (populated when calc_hardness=True)
            'thermal_conductivity_slack':  qe_props.get('thermal_conductivity_slack',  None),
            'elastic_anisotropy_zener':    qe_props.get('elastic_anisotropy_zener',    None),
            'peierls_stress_gpa':          qe_props.get('peierls_stress_gpa',          None),
            # Tier 2: carrier physics (always populated for non-metals)
            'gap_type':                    qe_props.get('gap_type',                    None),
            'carrier_lifetime_radiative':  qe_props.get('carrier_lifetime_radiative',  None),
            'recombination_coefficient_B': qe_props.get('recombination_coefficient_B', None),
            'intrinsic_carrier_density':   qe_props.get('intrinsic_carrier_density',   None),
            # Tier 3: deformation potential mobility (populated when calc_mobility=True)
            'deformation_potential_e':     qe_props.get('deformation_potential_e',     None),
            'deformation_potential_h':     qe_props.get('deformation_potential_h',     None),
            'mobility_e':                  qe_props.get('mobility_e',                  None),
            'mobility_h':                  qe_props.get('mobility_h',                  None),
            # Tier 4: thermodynamic stability (populated when calc_hull_distance=True)
            'e_above_hull':                qe_props.get('e_above_hull',                None),
            'is_thermodynamically_stable': qe_props.get('is_thermodynamically_stable', None),
            'error':             qe_props.get('error'),
        }
