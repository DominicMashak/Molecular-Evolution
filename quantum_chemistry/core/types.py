#!/usr/bin/env python3
"""
Type definitions and data classes for molecular property calculations.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
import numpy as np


class CalculatorType(Enum):
    """Available calculator backends."""
    DFT = "dft"
    XTB = "xtb"
    CC = "cc"
    MOPAC = "mopac" 
    SEMIEMPIRICAL = "semiempirical"


class MethodType(Enum):
    """Available hyperpolarizability calculation methods."""
    EMPIRICAL = "empirical"
    FINITE_FIELD = "finite_field"
    FULL_TENSOR = "full_tensor"
    CPHF = "cphf"


class DFTFunctional(Enum):
    """Common DFT functionals."""
    HF = "HF"
    B3LYP = "B3LYP"
    PBE = "PBE"
    PBE0 = "PBE0"
    CAM_B3LYP = "CAM-B3LYP"
    WB97X = "wB97X"
    M06_2X = "M06-2X"
    BLYP = "BLYP"


class XTBMethod(Enum):
    """xTB method variants."""
    GFN0 = "GFN0-xTB"
    GFN1 = "GFN1-xTB"
    GFN2 = "GFN2-xTB"
    GFN_FF = "GFN-FF"


@dataclass
class Molecule:
    """Container for molecular data."""
    smiles: str
    atomic_numbers: np.ndarray
    positions: np.ndarray  # In Angstrom
    charge: int = 0
    spin: int = 1  # Multiplicity
    formula: Optional[str] = None
    rdkit_mol: Optional[Any] = None  # RDKit Mol object
    
    @property
    def num_atoms(self) -> int:
        """Number of atoms in molecule."""
        return len(self.atomic_numbers)
    
    @property
    def num_electrons(self) -> int:
        """Total number of electrons."""
        return sum(self.atomic_numbers) - self.charge


@dataclass
class CalculatorConfig:
    """Configuration for calculator setup."""
    calculator_type: CalculatorType
    # DFT specific
    functional: Optional[str] = None
    basis: Optional[str] = None
    # xTB specific
    xtb_method: Optional[str] = None
    # Semiempirical specific
    se_method: Optional[str] = None
    # Common
    solvent: Optional[str] = None
    verbose: bool = False
    max_iterations: int = 250
    convergence_threshold: float = 1e-9
    temperature: float = 300.0  # Kelvin
    # New: carry dielectric and other flags/properties through the config
    dielectric: Optional[float] = None
    debug_mopac: bool = False
    properties: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for passing to calculators."""
        config = {
            'verbose': self.verbose,
            'max_iterations': self.max_iterations,
            'convergence_threshold': self.convergence_threshold,
            'temperature': self.temperature,
        }
        if self.solvent:
            config['solvent'] = self.solvent
        if self.dielectric is not None:
            config['dielectric'] = self.dielectric
        if self.functional:
            config['functional'] = self.functional
        if self.basis:
            config['basis'] = self.basis
        if self.xtb_method:
            config['xtb_method'] = self.xtb_method
        if self.se_method:
            config['se_method'] = self.se_method
        # Pass through optional flags and properties
        config['debug_mopac'] = self.debug_mopac
        if self.properties is not None:
            config['properties'] = self.properties
        return config


@dataclass
class MethodConfig:
    """Configuration for hyperpolarizability method."""
    method_type: MethodType
    field_strength: float = 0.001  # For finite field methods
    verbose: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for passing to methods."""
        return {
            'field_strength': self.field_strength,
            'verbose': self.verbose,
        }


@dataclass
class MoleculeResult:
    """Complete result for a single molecule calculation."""
    smiles: str
    formula: Optional[str] = None
    method: str = ""
    beta_vec: Optional[float] = None
    beta_xxx: Optional[float] = None
    beta_yyy: Optional[float] = None
    beta_zzz: Optional[float] = None
    beta_mean: Optional[float] = None
    dipole_moment: Optional[float] = None
    homo_lumo_gap: Optional[float] = None
    total_energy: Optional[float] = None
    transition_dipole: Optional[float] = None 
    oscillator_strength: Optional[float] = None 
    gamma: Optional[float] = None 
    alpha_mean: Optional[float] = None 
    wall_time: float = 0.0
    error: Optional[str] = None
    # Additional tensor components for full tensor method
    beta_tensor: Optional[List[List[List[float]]]] = None
    alpha_tensor: Optional[List[List[float]]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DataFrame/CSV export."""
        result = {}
        for key, value in self.__dict__.items():
            if value is not None:
                if isinstance(value, np.ndarray):
                    # Skip tensor arrays for CSV output
                    if key not in ['beta_tensor', 'alpha_tensor']:
                        result[key] = value
                else:
                    result[key] = value
        return result


@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    input_file: str
    output_file: str
    max_molecules: Optional[int] = None
    save_interval: int = 10
    skip_errors: bool = True
    parallel: bool = False
    n_workers: int = 1