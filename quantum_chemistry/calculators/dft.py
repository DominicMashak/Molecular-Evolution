#!/usr/bin/env python3
"""
DFT calculator implementation using PySCF.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
import sys
import io
from contextlib import redirect_stdout, redirect_stderr
from pyscf import lib

try:
    from pyscf import gto, scf, dft
    PYSCF_AVAILABLE = True
except ImportError:
    PYSCF_AVAILABLE = False

try:
    from core.base import Calculator
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.base import Calculator

from calculators.setup import setup_molecule, get_system_max_memory


class DFTCalculator(Calculator):
    """
    DFT calculator using PySCF backend.
    """
    
    def __init__(self, functional: str = 'B3LYP', basis: str = '6-31G*', 
                 dielectric: Optional[float] = None, verbose: bool = False, max_memory: int = None, method: str = 'single_point', **kwargs):
        """
        Initialize DFT calculator.
        
        Args:
            functional: DFT functional (e.g., 'B3LYP', 'PBE', 'PBE0')
            basis: Basis set (e.g., '6-31G*', 'def2-SVP')
            dielectric: Dielectric constant for implicit solvation
            verbose: Print detailed output
            max_memory: Maximum memory for PySCF in MB
            method: Calculation method ('single_point' for energy, 'finite_field' for beta)
            **kwargs: Additional PySCF parameters
        """
        super().__init__(verbose=verbose, **kwargs)
        # Use system memory if max_memory not provided
        if max_memory is None:
            max_memory = get_system_max_memory()
        
        if not PYSCF_AVAILABLE:
            raise ImportError("PySCF is required for DFT calculations. Install with: pip install pyscf")
        
        self.functional = functional.upper()
        self.basis = basis
        self.dielectric = dielectric
        self.max_memory = max_memory
        self.method = method.lower()
        
        # Map common basis set notations to PySCF format
        self.basis_map = {
            '6-31G*': '6-31g(d)',
            '6-31G**': '6-31g(d,p)',
            '6-311G*': '6-311g(d)',
            '6-311G**': '6-311g(d,p)',
            '6-31+G*': '6-31+g(d)',
            '6-311++G**': '6-311++g(d,p)',
            'def2-SVP': 'def2-svp',
            'def2-TZVP': 'def2-tzvp',
            'def2-TZVPP': 'def2-tzvpp',
            'cc-pVDZ': 'cc-pvdz',
            'cc-pVTZ': 'cc-pvtz',
            'aug-cc-pVDZ': 'aug-cc-pvdz',
        }
    
    def setup(self, atomic_numbers: np.ndarray, positions: np.ndarray,
              charge: int = 0, spin: int = 1) -> gto.Mole:
        """
        Setup PySCF molecule object.
        
        Args:
            atomic_numbers: Array of atomic numbers
            positions: Array of atomic positions in Angstrom
            charge: Molecular charge
            spin: Spin multiplicity (2S+1)
            
        Returns:
            PySCF Mole object
        """
        # Convert atomic numbers to symbols
        from periodictable import elements
        atomic_symbols = [elements[int(z)].symbol for z in atomic_numbers]
        
        # Build atom string for PySCF
        atom_str = []
        for i, symbol in enumerate(atomic_symbols):
            x, y, z = positions[i]
            atom_str.append(f"{symbol} {x:.6f} {y:.6f} {z:.6f}")
        
        # Create molecule object
        mol = gto.Mole()
        mol.atom = '\n'.join(atom_str)
        mol.basis = self.basis
        mol.charge = charge
        mol.spin = spin - 1  # PySCF uses number of unpaired electrons
        mol.unit = 'Angstrom'
        mol.verbose = 0 if not self.verbose else 4
        mol.max_memory = self.max_memory
        
        # PCM setup using dielectric constant
        if self.dielectric and self.dielectric > 1.0:
            from pyscf import solvent
            mol = solvent.PCM(mol, epsilon=self.dielectric)
        
        mol.build()
        
        return mol
    
    def single_point(self, atomic_numbers: np.ndarray, positions: np.ndarray,
                    charge: int = 0, spin: int = 1,
                    electric_field: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Perform DFT single point calculation.
        
        Args:
            atomic_numbers: Array of atomic numbers
            positions: Array of atomic positions in Angstrom
            charge: Molecular charge
            spin: Spin multiplicity (2S+1)
            electric_field: External electric field in a.u. [Ex, Ey, Ez]
            
        Returns:
            Dictionary with calculation results
        """
        # Setup molecule
        mol = self.setup(atomic_numbers, positions, charge, spin)
        
        # Choose SCF method
        if self.functional in ['HF', 'HARTREE-FOCK']:
            mf = scf.RHF(mol) if mol.spin == 0 else scf.UHF(mol)
        else:
            mf = dft.RKS(mol) if mol.spin == 0 else dft.UKS(mol)
            mf.xc = self.functional
        
        # Add electric field if specified
        if electric_field is not None and np.any(electric_field):
            # Get dipole integrals
            h1 = mol.intor('int1e_r', comp=3)
            field_au = np.array(electric_field)
            
            # Add field contribution to Hamiltonian
            original_hcore = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
            mf.get_hcore = lambda *args: original_hcore + np.einsum('i,ijk->jk', field_au, h1)
        
        # Set convergence parameters
        mf.conv_tol = self.config.get('convergence_threshold', 1e-9)
        mf.conv_tol_grad = self.config.get('conv_tol_grad', 1e-6)
        mf.max_cycle = self.config.get('max_iterations', 100)
        
        # Run calculation (suppress output if not verbose)
        if not self.verbose:
            f = io.StringIO()
            with redirect_stdout(f), redirect_stderr(f):
                energy = mf.kernel()
        else:
            energy = mf.kernel()
        
        # Calculate dipole moment
        dip_ints = mol.intor('int1e_r', comp=3)
        dm = mf.make_rdm1()
        elec_dip = -np.einsum('xij,ji->x', dip_ints, dm)
        
        # Add nuclear contribution
        charges = mol.atom_charges()
        coords_bohr = mol.atom_coords()  # Already in Bohr
        nuc_dip = np.einsum('i,ix->x', charges, coords_bohr)
        
        # Total dipole
        dipole = elec_dip + nuc_dip
        
        return {
            'energy': energy,
            'dipole': np.linalg.norm(dipole),  # Magnitude
            'dipole_vector': dipole,  # Vector for full_tensor
            'converged': mf.converged,
            'mf': mf,  # Keep for orbital info extraction
            'mol': mol
        }
    
    def calculate_polarizability(self, atomic_numbers: np.ndarray, positions: np.ndarray,
                                 charge: int = 0, spin: int = 1, field_strength: float = 0.001) -> float:
        """
        Calculate mean polarizability using finite field method.
        
        Args:
            atomic_numbers: Array of atomic numbers
            positions: Array of atomic positions in Angstrom
            charge: Molecular charge
            spin: Spin multiplicity (2S+1)
            field_strength: Electric field strength for perturbation (a.u.)
            
        Returns:
            Mean polarizability (a.u.)
        """
        # Calculate dipole at zero field and with fields along each axis
        result_0 = self.single_point(atomic_numbers, positions, charge, spin)
        mu_0 = result_0['dipole']
        
        alpha_tensor = np.zeros((3, 3))
        
        # Calculate polarizability tensor components
        for i in range(3):
            # Positive field
            field_pos = np.zeros(3)
            field_pos[i] = field_strength
            result_pos = self.single_point(atomic_numbers, positions, charge, spin, electric_field=field_pos)
            mu_pos = result_pos['dipole']
            
            # Negative field
            field_neg = np.zeros(3)
            field_neg[i] = -field_strength
            result_neg = self.single_point(atomic_numbers, positions, charge, spin, electric_field=field_neg)
            mu_neg = result_neg['dipole']
            
            # Calculate polarizability using central difference
            # alpha_ij = d(mu_i)/dE_j
            alpha_tensor[:, i] = (mu_pos - mu_neg) / (2 * field_strength)
        
        # Mean polarizability is the trace divided by 3
        alpha_mean = np.trace(alpha_tensor) / 3.0
        
        return alpha_mean
    
    def calculate_gamma(self, atomic_numbers: np.ndarray, positions: np.ndarray,
                       charge: int = 0, spin: int = 1, field_strength: float = 0.001) -> float:
        """
        Calculate mean second hyperpolarizability (gamma) using finite field method.
        
        Args:
            atomic_numbers: Array of atomic numbers
            positions: Array of atomic positions in Angstrom
            charge: Molecular charge
            spin: Spin multiplicity (2S+1)
            field_strength: Electric field strength for perturbation (a.u.)
            
        Returns:
            Mean gamma (a.u.)
        """
        # For gamma, we need the fourth derivative of energy with respect to field
        # gamma_iiii = d^4E / (dE_i)^4
        # We'll calculate the diagonal components and average them
        
        gamma_components = []
        
        for i in range(3):
            # We need energies at 0, +h, -h, +2h, -2h
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
            
            # Fourth derivative using finite difference
            # f''''(0) ≈ [f(-2h) - 4f(-h) + 6f(0) - 4f(h) + f(2h)] / h^4
            gamma_ii = (E_m2 - 4*E_m1 + 6*E_0 - 4*E_p1 + E_p2) / (field_strength**4)
            gamma_components.append(gamma_ii)
        
        # Mean gamma (average of diagonal components)
        gamma_mean = np.mean(gamma_components)
        
        return gamma_mean
    
    def calculate_beta(self, atomic_numbers: np.ndarray, positions: np.ndarray,
                       charge: int = 0, spin: int = 1, field_strength: float = 0.001) -> Dict[str, Any]:
        """
        Calculate hyperpolarizability using full tensor finite field method.
        
        Args:
            atomic_numbers: Array of atomic numbers
            positions: Array of atomic positions in Angstrom
            charge: Molecular charge
            spin: Spin multiplicity (2S+1)
            field_strength: Electric field strength for perturbation (a.u.)
            
        Returns:
            Dictionary with beta results
        """
        from methods.full_tensor import FullTensorMethod
        
        # Use the full tensor method for accurate beta calculation
        method = FullTensorMethod(self, field_strength=field_strength, verbose=self.verbose)
        result = method.calculate(atomic_numbers, positions, charge, spin)
        
        # Extract relevant values
        beta_vec = result.beta_vec if result.beta_vec is not None else 0.0
        beta_mean = result.beta_mean if result.beta_mean is not None else 0.0
        beta_xxx = result.beta_xxx if result.beta_xxx is not None else 0.0
        beta_yyy = result.beta_yyy if result.beta_yyy is not None else 0.0
        beta_zzz = result.beta_zzz if result.beta_zzz is not None else 0.0
        
        # Calculate additional properties from reference calculation
        ref_result = self.single_point(atomic_numbers, positions, charge, spin)
        dipole_moment = np.linalg.norm(ref_result['dipole']) if ref_result['dipole'] is not None else None
        total_energy = ref_result['energy']
        
        # Get orbital information for HOMO-LUMO gap - FIX HERE
        homo, lumo = self.get_orbital_info(ref_result)
        homo_lumo_gap = (lumo - homo) * 27.211  # Convert to eV
        
        # Calculate transition dipole and oscillator strength
        try:
            trans_result = self.calculate_transition_dipole(atomic_numbers, positions, charge, spin, n_states=1)
            transition_dipole = trans_result['transition_dipole_magnitudes'][0] if trans_result['transition_dipole_magnitudes'] else None
            oscillator_strength = trans_result['oscillator_strengths'][0] if trans_result['oscillator_strengths'] else None
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not calculate transition dipole: {e}")
            transition_dipole = None
            oscillator_strength = None
        
        # Calculate polarizability (alpha) and second hyperpolarizability (gamma)
        try:
            alpha_mean = self.calculate_polarizability(atomic_numbers, positions, charge, spin, field_strength)
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not calculate polarizability: {e}")
            alpha_mean = None
        
        try:
            gamma = self.calculate_gamma(atomic_numbers, positions, charge, spin, field_strength)
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not calculate gamma: {e}")
            gamma = None
        
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
        """
        Calculate transition dipole moments using TD-DFT.
        
        Args:
            atomic_numbers: Array of atomic numbers
            positions: Array of atomic positions in Angstrom
            charge: Molecular charge
            spin: Spin multiplicity (2S+1)
            n_states: Number of excited states to compute
            
        Returns:
            Dictionary with transition dipole results
        """
        # First, perform ground-state DFT
        ground_result = self.single_point(atomic_numbers, positions, charge, spin)
        mf = ground_result['mf']
        
        if not mf.converged:
            raise RuntimeError("Ground-state SCF did not converge")
        
        # Perform TD-DFT
        from pyscf import tdscf
        if mf.spin == 0:
            td = tdscf.TDDFT(mf)
        
        td.nstates = n_states
        td.verbose = 0 if not self.verbose else 4
        
        # Run TD-DFT
        if not self.verbose:
            f = io.StringIO()
            with redirect_stdout(f), redirect_stderr(f):
                td.kernel()
        else:
            td.kernel()
        
        # Check convergence
        if not td.converged:
            raise RuntimeError("TD-DFT did not converge")
        
        # Extract properties using built-in methods
        transition_dipoles = td.transition_dipole()  # Array of [x, y, z] for each state
        oscillator_strengths = td.oscillator_strength()
        excitation_energies = td.e
        
        return {
            'transition_dipoles': transition_dipoles.tolist(),  # List of [x, y, z] vectors
            'transition_dipole_magnitudes': [np.linalg.norm(d) for d in transition_dipoles],
            'excitation_energies': excitation_energies.tolist(),
            'oscillator_strengths': oscillator_strengths.tolist(),
            'ground_energy': ground_result['energy']
        }
    
    # Use base class get_orbital_info
    
    @property
    def name(self) -> str:
        """Return calculator description."""
        return f"DFT/{self.functional}/{self.basis}"