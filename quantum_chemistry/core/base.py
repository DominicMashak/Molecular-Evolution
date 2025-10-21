#!/usr/bin/env python3
"""
Abstract base classes for molecular property calculations.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass


@dataclass
class CalculationResult:
    """Container for calculation results."""
    beta_vec: Optional[float] = None
    beta_xxx: Optional[float] = None
    beta_yyy: Optional[float] = None
    beta_zzz: Optional[float] = None
    beta_mean: Optional[float] = None
    beta_tensor: Optional[np.ndarray] = None
    dipole_moment: Optional[float] = None
    dipole_vector: Optional[np.ndarray] = None
    alpha_tensor: Optional[np.ndarray] = None
    alpha_mean: Optional[float] = None
    homo_lumo_gap: Optional[float] = None
    total_energy: Optional[float] = None
    transition_dipole: Optional[float] = None  
    oscillator_strength: Optional[float] = None 
    gamma: Optional[float] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, handling numpy arrays."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, np.ndarray):
                result[key] = value.tolist()
            else:
                result[key] = value
        return result


class Calculator(ABC):
    """
    Abstract base class for quantum chemical calculators.
    """
    
    def __init__(self, verbose: bool = False, **kwargs):
        """
        Initialize calculator.
        
        Args:
            verbose: Print detailed output
            **kwargs: Additional calculator-specific parameters
        """
        self.verbose = verbose
        self.config = kwargs
    
    @abstractmethod
    def setup(self, atomic_numbers: np.ndarray, positions: np.ndarray, 
              charge: int = 0, spin: int = 1) -> Any:
        """
        Setup the calculator for a specific molecule.
        
        Args:
            atomic_numbers: Array of atomic numbers
            positions: Array of atomic positions in Angstrom
            charge: Molecular charge
            spin: Spin multiplicity (2S+1)
            
        Returns:
            Calculator-specific molecule object
        """
        pass
    
    @abstractmethod
    def single_point(self, atomic_numbers: np.ndarray, positions: np.ndarray,
                    charge: int = 0, spin: int = 1, 
                    electric_field: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Perform single point calculation.
        
        Args:
            atomic_numbers: Array of atomic numbers
            positions: Array of atomic positions in Angstrom
            charge: Molecular charge
            spin: Spin multiplicity (2S+1)
            electric_field: External electric field in a.u. [Ex, Ey, Ez]
            
        Returns:
            Dictionary with at least:
                - energy: Total energy in a.u.
                - dipole: Dipole moment vector in a.u.
                - converged: Boolean indicating convergence
        """
        pass
    
    def get_orbital_info(self, calc_result: Any) -> Tuple[float, float]:
        """
        Generic HOMO/LUMO extraction for calculators that store 'mf' and 'mol' in calc_result.
        Subclasses can override if their format differs.
        """
        mf = calc_result.get('mf', None)
        mol = calc_result.get('mol', None)
        if mf is None or mol is None:
            return 0.0, 0.0
        mo_energy = mf.mo_energy
        mo_occ = mf.mo_occ
        if hasattr(mol, 'spin') and mol.spin == 0:
            # Restricted
            homo_idx = np.where(mo_occ > 0)[0][-1] if np.any(mo_occ > 0) else 0
            lumo_idx = homo_idx + 1
            homo = mo_energy[homo_idx] if homo_idx < len(mo_energy) else -0.5
            lumo = mo_energy[lumo_idx] if lumo_idx < len(mo_energy) else 0.0
        else:
            # Unrestricted
            try:
                mo_energy_a, mo_energy_b = mo_energy
                mo_occ_a, mo_occ_b = mo_occ
                homo_a = mo_energy_a[mo_occ_a > 0][-1] if any(mo_occ_a > 0) else -10.0
                homo_b = mo_energy_b[mo_occ_b > 0][-1] if any(mo_occ_b > 0) else -10.0
                homo = max(homo_a, homo_b)
                lumo_a = mo_energy_a[mo_occ_a == 0][0] if any(mo_occ_a == 0) else 10.0
                lumo_b = mo_energy_b[mo_occ_b == 0][0] if any(mo_occ_b == 0) else 10.0
                lumo = min(lumo_a, lumo_b)
            except Exception:
                homo = -0.5
                lumo = 0.0
        return homo, lumo
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return calculator name/description."""
        pass


class HyperpolarizabilityMethod(ABC):
    """
    Abstract base class for hyperpolarizability calculation methods.
    """
    
    def __init__(self, calculator: Calculator, verbose: bool = False, **kwargs):
        """
        Initialize method with a calculator backend.
        
        Args:
            calculator: Calculator instance to use
            verbose: Print detailed output
            **kwargs: Method-specific parameters
        """
        self.calculator = calculator
        self.verbose = verbose
        self.config = kwargs
    
    @abstractmethod
    def calculate(self, atomic_numbers: np.ndarray, positions: np.ndarray,
                 charge: int = 0, spin: int = 1, **kwargs) -> CalculationResult:
        """
        Calculate hyperpolarizability.
        
        Args:
            atomic_numbers: Array of atomic numbers
            positions: Array of atomic positions in Angstrom
            charge: Molecular charge
            spin: Spin multiplicity (2S+1)
            **kwargs: Method-specific parameters
            
        Returns:
            CalculationResult object with calculated properties
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return method name/description."""
        pass
    
    def _extract_tensor_components(self, beta_tensor: np.ndarray) -> Dict[str, float]:
        """
        Extract common tensor components and derived quantities.
        
        Args:
            beta_tensor: 3x3x3 hyperpolarizability tensor
            
        Returns:
            Dictionary with tensor components and derived values
        """
        components = {
            'beta_xxx': beta_tensor[0, 0, 0],
            'beta_yyy': beta_tensor[1, 1, 1],
            'beta_zzz': beta_tensor[2, 2, 2],
            'beta_xyy': beta_tensor[0, 1, 1],
            'beta_xzz': beta_tensor[0, 2, 2],
            'beta_yxx': beta_tensor[1, 0, 0],
            'beta_yzz': beta_tensor[1, 2, 2],
            'beta_zxx': beta_tensor[2, 0, 0],
            'beta_zyy': beta_tensor[2, 1, 1],
            'beta_xyz': beta_tensor[0, 1, 2],
        }
        
        # Calculate vector part
        beta_x = (beta_tensor[0, 0, 0] + beta_tensor[0, 1, 1] + beta_tensor[0, 2, 2]) / 3.0
        beta_y = (beta_tensor[1, 1, 1] + beta_tensor[1, 0, 0] + beta_tensor[1, 2, 2]) / 3.0
        beta_z = (beta_tensor[2, 2, 2] + beta_tensor[2, 0, 0] + beta_tensor[2, 1, 1]) / 3.0
        
        components['beta_x'] = beta_x
        components['beta_y'] = beta_y
        components['beta_z'] = beta_z
        components['beta_vec'] = np.sqrt(beta_x**2 + beta_y**2 + beta_z**2)
        
        # Calculate mean hyperpolarizability
        beta_mean = 0
        for i in range(3):
            beta_mean += beta_tensor[i, i, i]
            for j in range(3):
                if i != j:
                    beta_mean += 2 * beta_tensor[i, j, j]
        components['beta_mean'] = beta_mean / 5.0
        
        # Total norm
        components['beta_total'] = np.linalg.norm(beta_tensor)
        
        return components