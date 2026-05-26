"""
CrystalTcInterface — ALIGNN-based critical temperature (Tc) predictor.

Uses pretrained ALIGNN models from the JARVIS/NIST database, as employed by
the InvDesFlow superconductor inverse-design workflow (Han et al., Chinese
Physics Letters 42, 047301, 2025 | github.com/xqh19970407/InvDesFlow).

Models used (downloaded and cached automatically on first call):
  jv_supercon_tc_alignn        → Tc in Kelvin (BCS conventional SCs)
  jv_formation_energy_peratom_alignn → formation energy in eV/atom
  jv_mbj_bandgap               → MBJ band gap in eV (metallic if < 0.05)
  jv_supercon_debye_alignn     → Debye temperature in Kelvin

Superconductor screening filter (Wines et al. 2023):
  Tc > 5 K  AND  E_form < 0 eV/atom  AND  band_gap < 0.05 eV

The returned dict is archive-compatible with CrystalQEInterface — all keys
expected by algorithm archives are present so the two interfaces are drop-in
interchangeable.

Usage
-----
    from molev_utils.crystal_tc_interface import CrystalTcInterface

    ci = CrystalTcInterface(verbose=True)
    props = ci.calculate(slices_str, structure=pymatgen_structure)
    print(f"Tc = {props['tc']:.1f} K, candidate = {props['is_superconductor_candidate']}")

Requirements
------------
    pip install alignn jarvis-tools

Both packages are listed in environment.yml.  The 'alignn' package requires
DGL (dgl); DGL 2.1+ inference works without the optional graphbolt extension.
"""

from __future__ import annotations

import logging
import sys
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Ensure molev_utils is importable regardless of cwd
_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


# ---------------------------------------------------------------------------
# ALIGNN model names (JARVIS/NIST pretrained)
# ---------------------------------------------------------------------------

_MODEL_TC           = 'jv_supercon_tc_alignn'
_MODEL_EFORM        = 'jv_formation_energy_peratom_alignn'
_MODEL_BANDGAP      = 'jv_mbj_bandgap'
_MODEL_DEBYE        = 'jv_supercon_debye_alignn'

# Screening thresholds (Wines et al. 2023, same as InvDesFlow)
_TC_MIN_K     = 5.0    # Kelvin
_EFORM_MAX    = 0.0    # eV/atom  (negative = exothermic, stable)
_GAP_METALLIC = 0.05   # eV       (< threshold → treat as metallic)


# ---------------------------------------------------------------------------
# Archive-compatible zero dict
# ---------------------------------------------------------------------------

def _tc_error_props(slices_str: str, error: str) -> dict:
    """Return a zeroed, archive-compatible props dict with an error message."""
    return {
        # Identity
        'slices':                       slices_str,
        'smiles':                       None,
        # Archive binning shims
        'num_atoms':                    0,
        'num_bonds':                    0,
        # Structural
        'n_sites':                      0,
        'n_species':                    0,
        'volume':                       0.0,
        'density':                      0.0,
        'spacegroup':                   '? (0)',
        # ALIGNN Tc predictions
        'tc':                           0.0,
        'tc_formation_energy':          0.0,
        'tc_bandgap':                   0.0,
        'tc_debye_temperature':         0.0,
        'is_superconductor_candidate':  False,
        # Keys mirrored from CrystalQEInterface (archive compatibility)
        'total_energy':                 0.0,
        'formation_energy':             0.0,
        'bandgap':                      0.0,
        'bandgap_direct':               0.0,
        'vbm':                          0.0,
        'cbm':                          0.0,
        'is_metal':                     False,
        'effective_mass_e':             None,
        'effective_mass_h':             None,
        'band_structure':               None,
        'bulk_modulus':                 None,
        'shear_modulus':                None,
        'youngs_modulus':               None,
        'poisson_ratio':                None,
        'hardness_vickers':             None,
        'elastic_tensor':               None,
        'pugh_ratio':                   None,
        'cauchy_pressure':              None,
        'sound_velocity_longitudinal':  None,
        'sound_velocity_transverse':    None,
        'debye_temperature':            None,
        'fracture_toughness':           None,
        'is_dynamically_stable':        None,
        'n_imaginary_modes':            None,
        'phonon_frequencies_cm1':       None,
        'thermal_conductivity_slack':   None,
        'elastic_anisotropy_zener':     None,
        'peierls_stress_gpa':           None,
        'gap_type':                     None,
        'carrier_lifetime_radiative':   None,
        'recombination_coefficient_B':  None,
        'intrinsic_carrier_density':    None,
        'deformation_potential_e':      None,
        'deformation_potential_h':      None,
        'mobility_e':                   None,
        'mobility_h':                   None,
        'e_above_hull':                 None,
        'is_thermodynamically_stable':  None,
        'error':                        error,
    }


# ---------------------------------------------------------------------------
# Interface class
# ---------------------------------------------------------------------------

class CrystalTcInterface:
    """Predict superconducting Tc and screening properties via ALIGNN.

    Wraps the JARVIS/NIST pretrained ALIGNN models used in InvDesFlow.
    All models are downloaded from Figshare and cached locally on first use.

    Parameters
    ----------
    tc_threshold : float
        Minimum predicted Tc (K) to qualify as a superconductor candidate.
    verbose : bool
        Print progress messages during prediction.
    """

    def __init__(self, tc_threshold: float = _TC_MIN_K, verbose: bool = False):
        self.tc_threshold = tc_threshold
        self.verbose = verbose
        self._get_prediction = None  # lazy-loaded on first calculate()

    # ------------------------------------------------------------------
    # Lazy import helpers
    # ------------------------------------------------------------------

    def _load_alignn(self):
        """Import alignn.pretrained.get_prediction, suppressing DGL noise."""
        if self._get_prediction is not None:
            return self._get_prediction

        import warnings
        # DGL 2.1+ emits graphbolt warnings on import — suppress for inference
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            try:
                from alignn.pretrained import get_prediction
            except ImportError as exc:
                raise ImportError(
                    "The 'alignn' package is required for Tc prediction. "
                    "Install with:  pip install alignn jarvis-tools"
                ) from exc

        self._get_prediction = get_prediction
        return get_prediction

    @staticmethod
    def _structure_to_atoms(structure):
        """Convert a pymatgen Structure to a jarvis.core.atoms.Atoms object."""
        # Preferred: direct converter in jarvis-tools
        try:
            from jarvis.io.pymatgen.inputs import get_atoms_from_structure
            return get_atoms_from_structure(structure)
        except Exception:
            pass
        # Fallback: round-trip via CIF string
        from jarvis.core.atoms import Atoms as JAtoms
        cif_str = structure.to(fmt='cif')
        return JAtoms.from_cif(cif_str)

    # ------------------------------------------------------------------
    # Single-model prediction
    # ------------------------------------------------------------------

    def _predict(self, atoms, model_name: str) -> Optional[float]:
        """Run one ALIGNN model; return scalar float or None on failure."""
        get_prediction = self._load_alignn()
        try:
            result = get_prediction(
                model_name=model_name,
                atoms=atoms,
                cutoff=8,
                max_neighbors=12,
            )
            return float(result[0])
        except Exception as exc:
            logger.debug("[%s] prediction failed: %s", model_name, exc)
            if self.verbose:
                print(f"  WARNING [{model_name}]: {exc}")
            return None

    # ------------------------------------------------------------------
    # Public interface (drop-in replacement for CrystalQEInterface)
    # ------------------------------------------------------------------

    def calculate(self, slices_str: str, structure=None) -> dict:
        """Predict Tc and related properties for a SLICES crystal genotype.

        Parameters
        ----------
        slices_str : str
            SLICES crystal genotype string.
        structure : pymatgen.core.Structure, optional
            Pre-decoded structure.  If None, decoded from slices_str.

        Returns
        -------
        dict
            Archive-compatible property dict.  Key `tc` holds the predicted
            critical temperature in Kelvin; `is_superconductor_candidate` is
            True when all three screening criteria are satisfied.
        """
        # Decode if no structure supplied
        if structure is None:
            try:
                from slices_ops import SLICESMutator
                mutator = SLICESMutator()
                structure = mutator.to_structure(slices_str)
            except Exception as exc:
                return _tc_error_props(slices_str, f"SLICES decode error: {exc}")
            if structure is None:
                return _tc_error_props(slices_str, "SLICES decode failed")

        # Structural descriptors — no model required
        try:
            from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
            analyzer = SpacegroupAnalyzer(structure, symprec=0.1)
            sg_symbol = analyzer.get_space_group_symbol()
            sg_number = analyzer.get_space_group_number()
        except Exception:
            sg_symbol, sg_number = 'P1', 1

        n_sites = len(structure)
        n_species = len(set(str(s) for s in structure.species))

        # Convert structure once for all model calls
        try:
            atoms = self._structure_to_atoms(structure)
        except Exception as exc:
            return _tc_error_props(slices_str, f"Structure conversion failed: {exc}")

        if self.verbose:
            print(f"  Running ALIGNN predictions for {n_sites}-site, "
                  f"{n_species}-species crystal...")

        # Run the four ALIGNN models
        tc    = self._predict(atoms, _MODEL_TC)
        eform = self._predict(atoms, _MODEL_EFORM)
        gap   = self._predict(atoms, _MODEL_BANDGAP)
        debye = self._predict(atoms, _MODEL_DEBYE)

        # Screening filter (Wines et al. 2023 / InvDesFlow criteria)
        is_candidate = bool(
            tc    is not None and tc    >  self.tc_threshold and
            eform is not None and eform <  _EFORM_MAX        and
            gap   is not None and gap   <  _GAP_METALLIC
        )

        if self.verbose:
            print(f"  Tc={tc:.1f} K  Eform={eform:.3f} eV/at  "
                  f"gap={gap:.3f} eV  Debye={debye:.0f} K  "
                  f"candidate={'YES' if is_candidate else 'no'}")

        # Safe zero-fallback for failed predictions
        tc_val    = tc    if tc    is not None else 0.0
        eform_val = eform if eform is not None else 0.0
        gap_val   = gap   if gap   is not None else 0.0
        debye_val = debye if debye is not None else 0.0
        is_metal  = gap   is not None and gap < _GAP_METALLIC

        return {
            # Identity
            'slices':                       slices_str,
            'smiles':                       None,
            # Archive binning shims
            'num_atoms':                    n_sites,
            'num_bonds':                    0,
            # Structural
            'n_sites':                      n_sites,
            'n_species':                    n_species,
            'volume':                       round(structure.volume, 4),
            'density':                      round(structure.density, 4),
            'spacegroup':                   f"{sg_symbol} ({sg_number})",
            # ALIGNN Tc-specific outputs
            'tc':                           round(tc_val,    4),
            'tc_formation_energy':          round(eform_val, 6),
            'tc_bandgap':                   round(gap_val,   6),
            'tc_debye_temperature':         round(debye_val, 2),
            'is_superconductor_candidate':  is_candidate,
            # Mapped aliases for archive/objective compatibility
            'total_energy':                 0.0,
            'formation_energy':             round(eform_val, 6),
            'bandgap':                      round(gap_val,   6),
            'bandgap_direct':               0.0,
            'vbm':                          0.0,
            'cbm':                          0.0,
            'is_metal':                     is_metal,
            # Null-filled QE-only keys (archive compatibility)
            'effective_mass_e':             None,
            'effective_mass_h':             None,
            'band_structure':               None,
            'bulk_modulus':                 None,
            'shear_modulus':                None,
            'youngs_modulus':               None,
            'poisson_ratio':                None,
            'hardness_vickers':             None,
            'elastic_tensor':               None,
            'pugh_ratio':                   None,
            'cauchy_pressure':              None,
            'sound_velocity_longitudinal':  None,
            'sound_velocity_transverse':    None,
            'debye_temperature':            debye_val if debye else None,
            'fracture_toughness':           None,
            'is_dynamically_stable':        None,
            'n_imaginary_modes':            None,
            'phonon_frequencies_cm1':       None,
            'thermal_conductivity_slack':   None,
            'elastic_anisotropy_zener':     None,
            'peierls_stress_gpa':           None,
            'gap_type':                     None,
            'carrier_lifetime_radiative':   None,
            'recombination_coefficient_B':  None,
            'intrinsic_carrier_density':    None,
            'deformation_potential_e':      None,
            'deformation_potential_h':      None,
            'mobility_e':                   None,
            'mobility_h':                   None,
            'e_above_hull':                 None,
            'is_thermodynamically_stable':  None,
            'error':                        None,
        }
