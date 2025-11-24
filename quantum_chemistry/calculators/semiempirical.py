"""
Semi-empirical calculator implementation using PySCF (MINDO, MNDO, AM1, PM3) and MOPAC (PM7, PM6).
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
import subprocess
import tempfile
import os
import re
from rdkit import Chem
import sys

try:
    import pyscf
    from pyscf import gto, scf
    import pyscf.semiempirical as semiempirical
except ImportError:
    semiempirical = None

# Ensure the quantum_chemistry and its parent are in sys.path for imports
this_dir = os.path.dirname(os.path.abspath(__file__))
qc_dir = os.path.dirname(this_dir)
repo_dir = os.path.dirname(qc_dir)
sys.path.insert(0, qc_dir)
sys.path.insert(0, repo_dir)

from core.base import Calculator
from calculators.setup import setup_molecule, get_system_max_memory

SE_METHODS = [
    "MINDO3",
    "MNDO",
    "AM1",
    "PM3",
    "PM6",
    "PM7",
    "SCF",  
]

class SemiEmpiricalCalculator(Calculator):
    """
    Semi-empirical calculator using PySCF backend or MOPAC for PM6/PM7.
    """
    def __init__(self, method: str = "AM1", verbose: bool = False, max_memory: int = None, n_threads: int = None, **kwargs):
        super().__init__(verbose=verbose, **kwargs)
        if max_memory is None:
            max_memory = get_system_max_memory()
        self.method = kwargs.get('se_method', method).upper()
        self.verbose = verbose
        self.max_memory = max_memory
        self.n_threads = n_threads if n_threads is not None else 4
        self.config = kwargs

    def setup(self, atomic_numbers: np.ndarray, positions: np.ndarray, charge: int = 0, spin: int = 1):
        return setup_molecule(atomic_numbers, positions, charge, spin, self.max_memory)

    def calculate_beta(self, atomic_numbers: np.ndarray, positions: np.ndarray,
                       charge: int = 0, spin: int = 1, field_strength: float = 0.001) -> Dict[str, Any]:
        """Calculate beta using the appropriate method for the calculator."""
        if self.method in ("PM6", "PM7"):
            result = self.single_point(atomic_numbers, positions, charge, spin)
            return {
                'beta_vec': result.get('beta_mean', 0.0),
                'beta_mean': result.get('beta_mean', 0.0),
                'beta_xxx': result.get('beta_xxx', 0.0),
                'beta_yyy': result.get('beta_yyy', 0.0),
                'beta_zzz': result.get('beta_zzz', 0.0),
                'dipole_moment': np.linalg.norm(result['dipole']) if result.get('dipole') is not None else None,
                'homo_lumo_gap': None,
                'total_energy': result.get('energy', None),
                'transition_dipole': None,
                'oscillator_strength': None,
                'gamma': None,
                'alpha_mean': None,
            }
        else:
            from methods.full_tensor import FullTensorMethod
            method = FullTensorMethod(self, field_strength=field_strength, verbose=self.verbose)
            result = method.calculate(atomic_numbers, positions, charge, spin)
            beta_vec = result.beta_vec if result.beta_vec is not None else 0.0
            beta_mean = result.beta_mean if result.beta_mean is not None else 0.0
            beta_xxx = result.beta_xxx if result.beta_xxx is not None else 0.0
            beta_yyy = result.beta_yyy if result.beta_yyy is not None else 0.0
            beta_zzz = result.beta_zzz if result.beta_zzz is not None else 0.0
            dipole_moment = result.dipole_moment if hasattr(result, 'dipole_moment') else None
            total_energy = result.total_energy if hasattr(result, 'total_energy') else None
            homo_lumo_gap = result.homo_lumo_gap if hasattr(result, 'homo_lumo_gap') else None
            transition_dipole = getattr(result, 'transition_dipole', None)
            oscillator_strength = getattr(result, 'oscillator_strength', None)
            gamma = getattr(result, 'gamma', None)
            alpha_mean = getattr(result, 'alpha_mean', None)
            return {
                'beta_vec': beta_vec,
                'beta_xxx': beta_xxx,
                'beta_yyy': beta_yyy,
                'beta_zzz': beta_zzz,
                'beta_mean': beta_mean,
                'dipole_moment': dipole_moment,
                'homo_lumo_gap': homo_lumo_gap,
                'total_energy': total_energy,
                'transition_dipole': transition_dipole,
                'oscillator_strength': oscillator_strength,
                'gamma': gamma,
                'alpha_mean': alpha_mean,
            }

    def calculate_transition_dipole(self, atomic_numbers: np.ndarray, positions: np.ndarray,
                                    charge: int = 0, spin: int = 1, n_states: int = 5) -> Dict[str, Any]:
        """Transition dipole not supported."""
        return {'transition_dipoles': None, 'oscillator_strengths': None, 'excitation_energies': None}
    
    def get_orbital_info(self, calc_result: Any) -> Tuple[float, float]:
        mf = calc_result.get('mf', None)
        if mf is None:
            return 0.0, 0.0
        mo_energy = mf.mo_energy
        mo_occ = mf.mo_occ
        homo = max(mo_energy[mo_occ > 0.5]) if np.any(mo_occ > 0.5) else 0.0
        lumo = min(mo_energy[mo_occ < 0.5]) if np.any(mo_occ < 0.5) else 0.0
        return homo, lumo
    
    def calculate_gamma(self, atomic_numbers: np.ndarray, positions: np.ndarray,
                       charge: int = 0, spin: int = 1, field_strength: float = 0.001) -> float:
        """Calculate mean second hyperpolarizability (gamma) using finite field method."""
        gamma_components = []
        for i in range(3):
            field_0 = np.zeros(3)
            result_0 = self.single_point(atomic_numbers, positions, charge, spin, electric_field=field_0)
            E_0 = result_0['energy']
            field_p1 = np.zeros(3)
            field_p1[i] = field_strength
            result_p1 = self.single_point(atomic_numbers, positions, charge, spin, electric_field=field_p1)
            E_p1 = result_p1['energy']
            field_m1 = np.zeros(3)
            field_m1[i] = -field_strength
            result_m1 = self.single_point(atomic_numbers, positions, charge, spin, electric_field=field_m1)
            E_m1 = result_m1['energy']
            field_p2 = np.zeros(3)
            field_p2[i] = 2 * field_strength
            result_p2 = self.single_point(atomic_numbers, positions, charge, spin, electric_field=field_p2)
            E_p2 = result_p2['energy']
            field_m2 = np.zeros(3)
            field_m2[i] = -2 * field_strength
            result_m2 = self.single_point(atomic_numbers, positions, charge, spin, electric_field=field_m2)
            E_m2 = result_m2['energy']
            gamma_ii = (E_m2 - 4*E_m1 + 6*E_0 - 4*E_p1 + E_p2) / (field_strength**4)
            gamma_components.append(gamma_ii)
        gamma_mean = np.mean(gamma_components)
        return gamma_mean

    def calculate_alpha(self, *args, **kwargs):
        """Alpha not implemented."""
        return None
    
    def calculate_polarizability(self, atomic_numbers: np.ndarray, positions: np.ndarray,
                                 charge: int = 0, spin: int = 1, field_strength: float = 0.001) -> float:
        """Calculate mean polarizability using finite field method."""
        # Use dipole vectors when available (dipole_vector), otherwise fall back to 'dipole'
        result_0 = self.single_point(atomic_numbers, positions, charge, spin)
        mu0 = result_0.get('dipole_vector', result_0.get('dipole'))
        if mu0 is None:
            return 0.0

        alpha_tensor = np.zeros((3, 3))

        for i in range(3):
            field_pos = np.zeros(3)
            field_pos[i] = field_strength
            result_pos = self.single_point(atomic_numbers, positions, charge, spin, electric_field=field_pos)
            mu_pos = result_pos.get('dipole_vector', result_pos.get('dipole'))

            field_neg = np.zeros(3)
            field_neg[i] = -field_strength
            result_neg = self.single_point(atomic_numbers, positions, charge, spin, electric_field=field_neg)
            mu_neg = result_neg.get('dipole_vector', result_neg.get('dipole'))

            if mu_pos is None or mu_neg is None:
                alpha_tensor[:, i] = 0.0
                continue

            # Ensure numpy arrays for vector arithmetic
            mu_pos = np.asarray(mu_pos)
            mu_neg = np.asarray(mu_neg)

            # alpha_ij = d(mu_i)/dE_j
            alpha_tensor[:, i] = (mu_pos - mu_neg) / (2 * field_strength)

        alpha_mean = np.trace(alpha_tensor) / 3.0
        return alpha_mean

    def single_point(self, atomic_numbers: np.ndarray, positions: np.ndarray, charge: int = 0, spin: int = 1, electric_field: Optional[np.ndarray] = None) -> Dict[str, Any]:
        if self.method in ("PM6", "PM7"):
            mopac_method = self.method
            mopac_exe = os.environ.get('MOPAC_EXE', 'mopac')
            mopac_input_header = f"XYZ {mopac_method} RHF POLAR(E=(0.0)) GNORM=0.01 THREADS={self.n_threads*4}\n"
            xyz_lines = []
            for z, pos in zip(atomic_numbers, positions):
                symbol = Chem.GetPeriodicTable().GetElementSymbol(int(z))
                x, y, zc = pos
                xyz_lines.append(f"{symbol} {x:.6f} {y:.6f} {zc:.6f}")
            mopac_input = mopac_input_header + "\n".join(xyz_lines) + "\n"
            with tempfile.TemporaryDirectory() as tmpdir:
                mopac_file = os.path.join(tmpdir, 'mol.mop')
                with open(mopac_file, 'w') as f:
                    f.write(mopac_input)
                result = subprocess.run([mopac_exe, mopac_file], cwd=tmpdir, capture_output=True, text=True)
                output_file = mopac_file.replace('.mop', '.out')
                beta = None
                beta_pattern = r'AVERAGE BETA\s*\(SHG\)\s*VALUE AT\s*[\d.]+\s*EV\s*=\s*([-]?[\d.]+(?:[eE][+-]?\d+)?)\s*a\.u\.'
                energy = None
                dipole = None
                dipole_pattern = r"DIPOLE\s*=\s*([-]?\d+\.\d+)\s*DEBYE"
                energy_pattern = r"FINAL HEAT OF FORMATION\s*=\s*([-]?\d+\.\d+)"
                if os.path.exists(output_file):
                    with open(output_file, 'r') as f:
                        content = f.read()
                    dipole_match = re.search(dipole_pattern, content, re.IGNORECASE)
                    if dipole_match:
                        dipole = float(dipole_match.group(1)) / 2.542
                    energy_match = re.search(energy_pattern, content, re.IGNORECASE)
                    if energy_match:
                        energy = float(energy_match.group(1))
                    match = re.search(beta_pattern, content, re.IGNORECASE)
                    if match:
                        beta = float(match.group(1))
                if beta is None or self.config.get('debug_mopac', False):
                    print("MOPAC output content:")
                    print(content)
                    return {
                        'energy': 0.0,
                        'dipole': None,
                        'converged': False,
                        'mf': None,
                        'mol': None,
                        'error': 'Failed to extract beta from MOPAC output'
                    }
                return {
                    'energy': energy or beta,
                    'dipole': dipole,
                    'converged': True,
                    'mf': None,
                    'mol': None,
                    'beta_vec': beta,
                    'beta_xxx': beta,
                    'beta_yyy': beta,
                    'beta_zzz': beta,
                    'beta_mean': beta,
                    'beta_tensor': None,
                    'alpha_tensor': None,
                    'alpha_mean': None,
                    'homo_lumo_gap': None,
                    'total_energy': energy
                }

        if self.method == "SCF":
            # Plain SCF mode (RHF/UHF) using PySCF
            if 'gto' not in globals() or 'scf' not in globals():
                raise ImportError("PySCF is required for plain SCF calculations (se-method=SCF). Install pyscf.")
            mol = self.setup(atomic_numbers, positions, charge, spin)

            # Choose restricted or unrestricted SCF
            mf = scf.RHF(mol) if mol.spin == 0 else scf.UHF(mol)

            # Add electric field if specified
            if electric_field is not None and np.any(electric_field):
                try:
                    h1 = mol.intor('int1e_r', comp=3)
                    field_au = np.array(electric_field)
                    original_hcore = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
                    mf.get_hcore = lambda *args: original_hcore + np.einsum('i,ijk->jk', field_au, h1)
                except Exception as e:
                    if self.verbose:
                        print(f"  Warning: could not apply electric field to SCF hcore: {e}")

            mf.conv_tol = self.config.get('convergence_threshold', 1e-9)
            mf.max_cycle = self.config.get('max_iterations', 100)

            energy = mf.kernel()
            try:
                dip_ints = mol.intor('int1e_r', comp=3)
                dm = mf.make_rdm1()
                elec_dip = -np.einsum('xij,ji->x', dip_ints, dm)
                charges = mol.atom_charges()
                coords_bohr = mol.atom_coords()
                nuc_dip = np.einsum('i,ix->x', charges, coords_bohr)
                dipole_vec = elec_dip + nuc_dip
                dipole_mag = np.linalg.norm(dipole_vec)
            except Exception:
                # Fallback: if dipole integrals not compatible, return zero vector
                dipole_vec = np.zeros(3)
                dipole_mag = 0.0

            return {
                'energy': energy,
                'dipole': dipole_mag,
                'dipole_vector': dipole_vec,
                'converged': getattr(mf, 'converged', True),
                'mf': mf,
                'mol': mol
            }

        if semiempirical is None:
            raise ImportError("PySCF semiempirical module is not available. Please install pyscf with semiempirical support.")
        
        mol = self.setup(atomic_numbers, positions, charge, spin)

        method_class = None
        
        # MINDO3 is available as a function directly on the module
        if self.method == "MINDO3":
            method_class = getattr(semiempirical, 'MINDO3', None)
            if method_class and callable(method_class):
                if self.verbose:
                    print(f"  Found MINDO3 as function on semiempirical module")
            else:
                # Try the mindo3 submodule
                try:
                    from pyscf.semiempirical import mindo3
                    for cls_name in ['RMINDO3', 'MINDO3', 'UMINDO3']:
                        method_class = getattr(mindo3, cls_name, None)
                        if method_class:
                            if self.verbose:
                                print(f"  Found {cls_name} in mindo3 module")
                            break
                except ImportError:
                    method_class = None
        
        elif self.method == "AM1":
            try:
                from pyscf.semiempirical import am1
                # Use RAM1 for closed-shell, UAM1 for open-shell
                if mol.spin == 0:
                    method_class = getattr(am1, 'RAM1', None)
                    if self.verbose and method_class:
                        print(f"  Using RAM1 (Restricted AM1) for closed-shell system")
                else:
                    method_class = getattr(am1, 'UAM1', None)
                    if self.verbose and method_class:
                        print(f"  Using UAM1 (Unrestricted AM1) for open-shell system")
            except ImportError:
                method_class = None
        
        elif self.method in ["PM3", "MNDO"]:
            try:
                import importlib
                mod_name = self.method.lower()
                mod = importlib.import_module(f"pyscf.semiempirical.{mod_name}")
                method_class = getattr(mod, self.method, None) or \
                             getattr(mod, f"R{self.method}", None) or \
                             getattr(mod, f"U{self.method}", None)
            except ImportError:
                method_class = None

        if method_class is None:
            raise ImportError(
                f"PySCF semiempirical method '{self.method}' is not available in your installation.\n"
                f"Available methods: AM1, MINDO3\n"
                f"For PM6/PM7: Use --calculator semiempirical --se-method PM6/PM7 (requires MOPAC)\n"
                f"For PM3/MNDO: These may require a different PySCF version or installation"
            )

        if self.verbose:
            print(f"  Using method: {method_class.__name__ if hasattr(method_class, '__name__') else method_class}")

        try:
            # For MINDO3 function-based interface
            if self.method == "MINDO3" and not hasattr(method_class, '__init__'):
                if self.verbose:
                    print(f"  Calling MINDO3 function with mol")
                mf = method_class(mol)
            else:
                # Standard class-based interface (RAM1, UAM1, etc.)
                mf = method_class(mol)
            
            if self.verbose:
                print(f"  Initialized: {type(mf).__name__}")

            is_minimal_basis = self.method in ["MINDO3"]
            
            # Add electric field if specified  
            if electric_field is not None and np.any(electric_field):
                if is_minimal_basis:
                    # MINDO3 uses a special minimal basis - skip electric field modification
                    # The field won't work properly with the parametrized Hamiltonian
                    if self.verbose:
                        print(f"  Warning: Electric field not supported for {self.method}, continuing without field")
                else:
                    # Standard electric field application for AM1, PM3, etc.
                    try:
                        # Get dipole integrals  
                        h1 = mol.intor('int1e_r', comp=3)
                        field_au = np.array(electric_field)
                        
                        # Get original hcore - use mf.get_hcore to get the proper semiempirical Hamiltonian
                        if hasattr(mf, 'get_hcore'):
                            original_hcore = mf.get_hcore(mol)
                        else:
                            original_hcore = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
                        
                        # Verify shapes match
                        if h1.shape[1:] == original_hcore.shape:
                            # Create closure to capture values
                            def make_hcore_with_field(h0, h1_int, field):
                                def hcore_field(*args, **kwargs):
                                    return h0 + np.einsum('i,ijk->jk', field, h1_int)
                                return hcore_field
                            
                            mf.get_hcore = make_hcore_with_field(original_hcore, h1, field_au)
                            
                            if self.verbose:
                                print(f"  Applied electric field: {electric_field}")
                        else:
                            if self.verbose:
                                print(f"  Warning: Shape mismatch ({h1.shape} vs {original_hcore.shape}), skipping field")
                    except Exception as e:
                        if self.verbose:
                            print(f"  Warning: Could not apply electric field: {e}")

            mf.conv_tol = self.config.get('convergence_threshold', 1e-9)
            mf.max_cycle = self.config.get('max_iterations', 100)

            # Workaround for old PySCF versions that don't support s1e parameter
            # Patch get_init_guess to accept and ignore the s1e parameter
            original_get_init_guess = mf.get_init_guess
            def patched_get_init_guess(mol=None, key=None, s1e=None, **kwargs):
                # Call original without the s1e parameter
                if mol is None:
                    mol = mf.mol
                if key is None:
                    key = mf.init_guess
                return original_get_init_guess(mol, key)
            
            mf.get_init_guess = patched_get_init_guess
            
            if self.verbose:
                print(f"  Patched get_init_guess for compatibility")

            # Run calculation
            energy = mf.kernel()

            # Calculate dipole moment - check for basis compatibility
            if self.method in ["MINDO3", "AM1"]:
                # Skip dipole calculation for methods using minimal basis
                dipole_vec = np.zeros(3)
                dipole_mag = 0.0
            else:
                try:
                    dm = mf.make_rdm1()

                    # Try standard dipole integrals
                    try:
                        dip_ints = mol.intor('int1e_r', comp=3)
                        if dip_ints.shape[1:] == dm.shape:
                            elec_dip = -np.einsum('xij,ji->x', dip_ints, dm)
                            charges = mol.atom_charges()
                            coords_bohr = mol.atom_coords()
                            nuc_dip = np.einsum('i,ix->x', charges, coords_bohr)
                            dipole_vec = elec_dip + nuc_dip
                            dipole_mag = np.linalg.norm(dipole_vec)
                            if self.verbose:
                                print(f"  Dipole (standard): {dipole_mag:.6f} a.u.")
                        else:
                            raise ValueError("Dimension mismatch between dipole integrals and density matrix")

                    except Exception as e1:
                        # Fallback using Mulliken charges
                        if self.verbose:
                            print(f"  Standard dipole failed ({e1}), trying Mulliken approach")
                        try:
                            mulliken_charges = mf.mulliken_pop()[1]
                            coords_bohr = mol.atom_coords()
                            atomic_charges = mol.atom_charges()
                            nuc_dip = np.einsum('i,ix->x', atomic_charges, coords_bohr)
                            elec_dip = -np.einsum('i,ix->x', mulliken_charges, coords_bohr)
                            dipole_vec = nuc_dip + elec_dip
                            dipole_mag = np.linalg.norm(dipole_vec)
                            if self.verbose:
                                print(f"  Dipole (Mulliken): {dipole_mag:.6f} a.u.")
                        except Exception as e2:
                            # estimate or zero
                            if self.verbose:
                                print(f"  Mulliken approach failed ({e2}), using fallback estimate")
                            if electric_field is not None and np.any(electric_field):
                                dipole_vec = np.zeros(3)
                                dipole_mag = 0.0
                            else:
                                coords_bohr = mol.atom_coords()
                                charges = mol.atom_charges()
                                total_charge = np.sum(charges)
                                if abs(total_charge) > 0.01:
                                    center_pos = np.einsum('i,ix->x', charges, coords_bohr) / total_charge
                                    dipole_vec = center_pos * 0.1
                                    dipole_mag = np.linalg.norm(dipole_vec)
                                else:
                                    dipole_vec = np.zeros(3)
                                    dipole_mag = 0.0
                except Exception as e:
                    if self.verbose:
                        print(f"  All dipole calculation methods failed: {e}")
                    dipole_vec = np.zeros(3)
                    dipole_mag = 0.0

            # Return result structure
            return {
                'energy': energy,
                'dipole': dipole_mag,
                'dipole_vector': dipole_vec,
                'converged': getattr(mf, 'converged', True),
                'mf': mf,
                'mol': mol
            }

        except Exception as e:
            if self.verbose:
                print(f"  Error during {self.method} calculation: {e}")
                import traceback
                traceback.print_exc()
            raise RuntimeError(f"Semiempirical {self.method} calculation failed: {e}")

    @property
    def name(self) -> str:
        return f"SemiEmpirical/{self.method}"