#!/usr/bin/env python3
"""
xTB calculator using xtb-python (GFN2-xTB / GFN1-xTB).

Replaces the previous DFTB+-based implementation.  All calculations are run
in-process via the `xtb` Python bindings, which is faster and more reliable.

Electric fields are simulated with two large, opposing point charges placed
far from the molecule along each field axis (uniform-field approximation).

API is identical to the previous version so nothing else needs to change.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Any, Optional, Tuple

try:
    from core.base import Calculator
except ImportError:
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.base import Calculator


# ── Constants ─────────────────────────────────────────────────────────────────
_ANG2BOHR = 1.8897259886   # 1 Å in Bohr
_HARTREE2EV = 27.211386    # 1 Hartree in eV
_AU2DEBYE = 2.5417464      # dipole: 1 a.u. = 2.5417 Debye

# Distance of the external point charges used to simulate a uniform field.
# At 1000 Bohr (~529 Å) cross-axis contamination is < 0.1 % for a 10 Å molecule.
_FIELD_CHARGE_DIST = 1000.0   # Bohr


class XTBCalculator(Calculator):
    """
    xTB calculator using xtb-python bindings.

    Supports GFN0-xTB, GFN1-xTB, GFN2-xTB.
    Electric fields are simulated via external point charges for finite-field
    polarisability / hyperpolarisability calculations.
    """

    def __init__(self, xtb_method: str = 'GFN2-xTB', dielectric: Optional[float] = None,
                 verbose: bool = False, **kwargs):
        super().__init__(verbose=verbose, **kwargs)
        self.xtb_method = xtb_method.upper().replace('-', '')
        self.dielectric = dielectric

        # Map method name → xtb-python Param enum
        try:
            from xtb.interface import Param
            _param_map = {
                'GFN0XTB': Param.GFN0xTB,
                'GFN1XTB': Param.GFN1xTB,
                'GFN2XTB': Param.GFN2xTB,
            }
            key = self.xtb_method.replace('-', '').upper()
            if key not in _param_map:
                raise ValueError(
                    f"Unknown xTB method: {xtb_method}. "
                    f"Choose from: GFN0-xTB, GFN1-xTB, GFN2-xTB"
                )
            self._param = _param_map[key]
        except ImportError as e:
            raise RuntimeError(
                "xtb-python is required for the xTB calculator. "
                "Install with: conda install -c conda-forge xtb-python"
            ) from e

    # ── Core calculation ───────────────────────────────────────────────────────

    def single_point(self, atomic_numbers: np.ndarray, positions: np.ndarray,
                     charge: int = 0, spin: int = 1,
                     electric_field: Optional[np.ndarray] = None,
                     write_band: bool = False) -> Dict[str, Any]:
        """
        Run a GFN2-xTB single-point calculation.

        Parameters
        ----------
        atomic_numbers : (N,) int — atomic numbers
        positions      : (N, 3) float — Cartesian coordinates in ANGSTROM
        charge         : molecular charge (default 0)
        spin           : spin multiplicity 2S+1 (default 1 = singlet)
        electric_field : (3,) float array of field in ATOMIC UNITS [Ex, Ey, Ez].
                         Applied via external point charges (uniform-field approx.).
        write_band     : unused (kept for API compatibility)

        Returns
        -------
        dict with keys: energy, dipole, dipole_vector, orbital_energies,
                        orbital_occupations, converged
        """
        from xtb.interface import Calculator
        from xtb.libxtb import VERBOSITY_MUTED, VERBOSITY_FULL

        pos_bohr = np.asarray(positions, dtype=np.float64) * _ANG2BOHR
        nums = np.asarray(atomic_numbers, dtype=np.int32)
        uhf = max(0, spin - 1)

        calc = Calculator(self._param, nums, pos_bohr, charge=charge, uhf=uhf)
        calc.set_verbosity(VERBOSITY_FULL if self.verbose else VERBOSITY_MUTED)
        calc.set_accuracy(1.0)

        if self.dielectric and self.dielectric > 1.0:
            try:
                from xtb.interface import Solvent
                calc.set_solvent(Solvent.water)  # closest approximation
            except Exception:
                pass

        # ── Simulate uniform electric field with external point charges ────────
        if electric_field is not None:
            field = np.asarray(electric_field, dtype=np.float64)
            if np.any(np.abs(field) > 1e-12):
                ext_nums, ext_chg, ext_pos = self._field_charges(field)
                calc.set_external_charges(ext_nums, ext_chg, ext_pos)

        try:
            res = calc.singlepoint()
        except Exception as e:
            raise RuntimeError(f"xTB singlepoint failed: {e}")

        # ── Extract results ────────────────────────────────────────────────────
        energy = float(res.get_energy())
        dip_au = np.array(res.get_dipole(), dtype=float)   # atomic units
        dip_debye = dip_au * _AU2DEBYE

        evals = res.get_orbital_eigenvalues()   # Hartree
        occ   = res.get_orbital_occupations()

        result: Dict[str, Any] = {
            'energy': energy,
            'dipole': float(np.linalg.norm(dip_debye)),   # Debye magnitude
            'dipole_vector': dip_au,                       # a.u. vector (used by full_tensor)
            'orbital_energies': evals,
            'orbital_occupations': occ,
            'converged': True,
        }
        return result

    # ── Uniform-field helper ───────────────────────────────────────────────────

    @staticmethod
    def _field_charges(field: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build external point-charge arrays that produce a given uniform field.

        For field component Fi along axis i:
          +Q at -(d, 0, 0) along that axis and -Q at +(d, 0, 0)
          → field at origin = +Fi  (pointing in +i direction)

        Returns (numbers, charges, positions_bohr) for set_external_charges().
        """
        d = _FIELD_CHARGE_DIST
        nums_list, chg_list, pos_list = [], [], []

        for i, Fi in enumerate(field):
            if abs(Fi) < 1e-12:
                continue
            Q = Fi * d * d / 2.0
            pos_neg = [0.0, 0.0, 0.0]; pos_neg[i] = -d
            pos_pos = [0.0, 0.0, 0.0]; pos_pos[i] =  d
            nums_list += [1, 1]
            chg_list  += [Q, -Q]
            pos_list  += [pos_neg, pos_pos]

        return (
            np.array(nums_list, dtype=np.int32),
            np.array(chg_list,  dtype=np.float64),
            np.array(pos_list,  dtype=np.float64),
        )

    # ── Orbital info ───────────────────────────────────────────────────────────

    def get_orbital_info(self, calc_result: Dict[str, Any]) -> Tuple[float, float]:
        """
        Return (HOMO, LUMO) energies in HARTREE from a single_point result dict.

        Falls back to (0, 0) if orbital data is unavailable.
        """
        try:
            evals = calc_result.get('orbital_energies')
            occ   = calc_result.get('orbital_occupations')
            if evals is None or occ is None or len(evals) < 2:
                return 0.0, 0.0
            occupied = np.where(occ > 0.5)[0]
            if len(occupied) == 0:
                return 0.0, 0.0
            homo_idx = occupied.max()
            lumo_idx = homo_idx + 1
            if lumo_idx >= len(evals):
                return float(evals[homo_idx]), 0.0
            return float(evals[homo_idx]), float(evals[lumo_idx])
        except Exception:
            return 0.0, 0.0

    # ── Higher-order response properties ──────────────────────────────────────

    def calculate_polarizability(self, atomic_numbers, positions,
                                  charge=0, spin=1, field_strength=0.001) -> float:
        """Mean polarizability (a.u.) via finite-field central difference."""
        res0 = self.single_point(atomic_numbers, positions, charge, spin)
        mu0 = res0['dipole_vector']

        alpha_diag = []
        for i in range(3):
            F_p = np.zeros(3); F_p[i] =  field_strength
            F_m = np.zeros(3); F_m[i] = -field_strength
            mu_p = self.single_point(atomic_numbers, positions, charge, spin,
                                     electric_field=F_p)['dipole_vector']
            mu_m = self.single_point(atomic_numbers, positions, charge, spin,
                                     electric_field=F_m)['dipole_vector']
            alpha_diag.append((mu_p[i] - mu_m[i]) / (2 * field_strength))

        return float(np.mean(alpha_diag))

    def calculate_gamma(self, atomic_numbers, positions,
                         charge=0, spin=1, field_strength=0.001) -> float:
        """Mean second hyperpolarisability (a.u.) via fourth-order energy differences."""
        gamma_components = []
        for i in range(3):
            e0 = self.single_point(atomic_numbers, positions, charge, spin)['energy']
            def _e(f):
                F = np.zeros(3); F[i] = f
                return self.single_point(atomic_numbers, positions, charge, spin,
                                         electric_field=F)['energy']
            e_p1 = _e( field_strength)
            e_m1 = _e(-field_strength)
            e_p2 = _e( 2*field_strength)
            e_m2 = _e(-2*field_strength)
            gamma_components.append(
                (e_m2 - 4*e_m1 + 6*e0 - 4*e_p1 + e_p2) / field_strength**4
            )
        return float(np.mean(gamma_components))

    def calculate_beta(self, atomic_numbers, positions,
                        charge=0, spin=1, field_strength=0.001) -> Dict[str, Any]:
        """
        Full-tensor hyperpolarisability via finite-field method.
        Delegates to the shared full_tensor.FullTensorMethod.
        """
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from methods.full_tensor import FullTensorMethod

        method = FullTensorMethod(self, field_strength=field_strength, verbose=self.verbose)
        result = method.calculate(atomic_numbers, positions, charge, spin)

        beta_vec  = result.beta_vec  if result.beta_vec  is not None else 0.0
        beta_mean = result.beta_mean if result.beta_mean is not None else 0.0
        beta_xxx  = result.beta_xxx  if result.beta_xxx  is not None else 0.0
        beta_yyy  = result.beta_yyy  if result.beta_yyy  is not None else 0.0
        beta_zzz  = result.beta_zzz  if result.beta_zzz  is not None else 0.0

        ref = self.single_point(atomic_numbers, positions, charge, spin, write_band=True)
        homo, lumo = self.get_orbital_info(ref)
        homo_lumo_gap = (lumo - homo) * _HARTREE2EV

        try:
            alpha_mean = self.calculate_polarizability(
                atomic_numbers, positions, charge, spin, field_strength)
        except Exception:
            alpha_mean = None

        try:
            gamma = self.calculate_gamma(
                atomic_numbers, positions, charge, spin, field_strength)
        except Exception:
            gamma = None

        return {
            'beta_vec':     beta_vec,
            'beta_xxx':     beta_xxx,
            'beta_yyy':     beta_yyy,
            'beta_zzz':     beta_zzz,
            'beta_mean':    beta_mean,
            'dipole_moment': float(ref['dipole']),
            'homo_lumo_gap': homo_lumo_gap,
            'total_energy':  ref['energy'],
            'transition_dipole': None,
            'oscillator_strength': None,
            'gamma':         gamma,
            'alpha_mean':    alpha_mean,
        }

    def calculate_transition_dipole(self, *args, **kwargs) -> Dict[str, Any]:
        """TD-DFT not available for xTB. Returns None values for API compatibility."""
        return {
            'transition_dipoles': None,
            'transition_dipole_magnitudes': None,
            'excitation_energies': None,
            'oscillator_strengths': None,
            'ground_energy': None,
        }

    def setup(self, atomic_numbers, positions, charge=0, spin=1):
        """No-op setup (xtb-python reads data directly)."""
        return atomic_numbers, positions, charge, spin

    @property
    def name(self) -> str:
        return f"xTB/{self.xtb_method}"
