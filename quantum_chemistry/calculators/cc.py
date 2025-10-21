"""
Coupled Cluster (CC) calculator implementation and definitions.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
try:
	from pyscf import gto, scf, cc
	PYSCF_AVAILABLE = True
except ImportError:
	PYSCF_AVAILABLE = False

try:
	from core.base import Calculator
except ImportError:
	import sys, os
	sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
	from core.base import Calculator
from pyscf import lib
from calculators.setup import setup_molecule, get_system_max_memory

# Supported CC functionals (methods)
CC_FUNCTIONALS = [
	"CCSD",
	"CCSD(T)",
]

# Supported basis sets
CC_BASIS_SETS = [
	"cc-pVDZ",
	"cc-pVTZ",
	"cc-pVQZ",
	"aug-cc-pVDZ",
	"aug-cc-pVTZ",
	"def2-TZVP",
]

def get_cc_functionals():
	return CC_FUNCTIONALS

def get_cc_basis_sets():
	return CC_BASIS_SETS

class CCCalculator(Calculator):
	"""
	Coupled Cluster calculator using PySCF backend.
	"""
	def __init__(self, functional: str = "CCSD", basis: str = "cc-pVDZ", dielectric: Optional[float] = None, verbose: bool = False, max_memory: int = None, **kwargs):
		super().__init__(verbose=verbose, **kwargs)
		if max_memory is None:
			max_memory = get_system_max_memory()
		if not PYSCF_AVAILABLE:
			raise ImportError("PySCF is required for CC calculations. Install with: pip install pyscf")
		self.functional = functional.upper()
		self.basis = basis
		self.dielectric = dielectric
		self.verbose = verbose
		self.max_memory = max_memory
		self.config = kwargs

	def setup(self, atomic_numbers: np.ndarray, positions: np.ndarray, charge: int = 0, spin: int = 1):
		mol = setup_molecule(atomic_numbers, positions, charge, spin, self.max_memory)
		mol.basis = self.basis
		# PCM setup using dielectric constant
		if self.dielectric and self.dielectric > 1.0:
			from pyscf import solvent
			mol = solvent.PCM(mol, epsilon=self.dielectric)
		return mol

	def single_point(self, atomic_numbers: np.ndarray, positions: np.ndarray, charge: int = 0, spin: int = 1, electric_field: Optional[np.ndarray] = None) -> Dict[str, Any]:
		mol = self.setup(atomic_numbers, positions, charge, spin)
		mf = scf.RHF(mol) if mol.spin == 0 else scf.UHF(mol)
		mf.kernel()
		if self.functional == "CCSD":
			cc_obj = cc.CCSD(mf)
			cc_obj.kernel()
			energy = cc_obj.e_tot
		elif self.functional == "CCSD(T)":
			cc_obj = cc.CCSD(mf)
			cc_obj.kernel()
			energy = cc_obj.e_tot + cc_obj.ccsd_t()
		else:
			raise ValueError(f"Unsupported CC functional: {self.functional}")
		dipole = np.zeros(3)
		return {
			'energy': energy,
			'dipole': dipole,
			'converged': True,
			'mf': mf,
			'mol': mol,
			'cc': cc_obj
		}

	def calculate_beta(self, atomic_numbers: np.ndarray, positions: np.ndarray,
	                   charge: int = 0, spin: int = 1, field_strength: float = 0.001) -> Dict[str, Any]:
		"""Calculate hyperpolarizability using finite field on CCSD (expensive)."""
		# placeholder implementation
		raise NotImplementedError("Beta calculation for CC not implemented. Use DFT for accurate beta.")
	
	def calculate_transition_dipole(self, atomic_numbers: np.ndarray, positions: np.ndarray,
	                                charge: int = 0, spin: int = 1, n_states: int = 5) -> Dict[str, Any]:
		"""Transition dipole not supported in PySCF CC."""
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
	
	# placeholders for other properties
	def calculate_gamma(self, *args, **kwargs):
		"""Gamma not implemented for CC."""
		return None
	
	def calculate_alpha(self, *args, **kwargs):
		"""Alpha not implemented for CC."""
		return None

	@property
	def name(self) -> str:
		return f"CC/{self.functional}/{self.basis}"

	@property
	def name(self) -> str:
		return f"CC/{self.functional}/{self.basis}"