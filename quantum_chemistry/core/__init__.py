"""
Initialization files for Calculator Framework.

Core components for molecular property calculations.
"""

from core.base import Calculator, HyperpolarizabilityMethod, CalculationResult
from core.processor import Processor
from core.types import (
    Molecule,
    MoleculeResult,
    CalculatorConfig,
    MethodConfig,
    BatchConfig,
    CalculatorType,
    MethodType,
    DFTFunctional,
    XTBMethod,
)

__all__ = [
    # Base classes
    'Calculator',
    'HyperpolarizabilityMethod',
    'CalculationResult',
    # Processor
    'Processor',
    # Data types
    'Molecule',
    'MoleculeResult',
    'CalculatorConfig',
    'MethodConfig',
    'BatchConfig',
    # Enums
    'CalculatorType',
    'MethodType',
    'DFTFunctional',
    'XTBMethod',
]


"""
Quantum chemistry calculator backends.
"""

calculators = {}

try:
    from calculators.dft import DFTCalculator
    calculators['dft'] = DFTCalculator
except ImportError:
    pass

try:
    from calculators.xtb import XTBCalculator
    calculators['xtb'] = XTBCalculator
except ImportError:
    pass

try:
    from calculators.cc import CCCalculator
    calculators['cc'] = CCCalculator
except ImportError:
    pass

def get_calculator(name: str):
    """
    Get calculator class by name.
    
    Args:
        name: Calculator name ('dft' or 'xtb')
        
    Returns:
        Calculator class
        
    Raises:
        ValueError: If calculator not found or not available
    """
    if name.lower() not in calculators:
        available = list(calculators.keys())
        if available:
            raise ValueError(f"Calculator '{name}' not available. Available: {available}")
        else:
            raise ValueError("No calculators available. Install pyscf or tblite.")
    return calculators[name.lower()]

__all__ = ['get_calculator'] + list(calculators.keys())


"""
Hyperpolarizability calculation methods.
"""

from methods.finite_field import FiniteFieldMethod
from methods.full_tensor import FullTensorMethod
from methods.cphf import CPHFMethod

# Method registry
methods = {
    'finite_field': FiniteFieldMethod,
    'full_tensor': FullTensorMethod,
    'cphf': CPHFMethod,
}

def get_method(name: str):
    """
    Get method class by name.
    
    Args:
        name: Method name
        
    Returns:
        Method class
        
    Raises:
        ValueError: If method not found
    """
    if name.lower() not in methods:
        available = list(methods.keys())
        raise ValueError(f"Method '{name}' not found. Available: {available}")
    return methods[name.lower()]

__all__ = [
    'FiniteFieldMethod',
    'FullTensorMethod',
    'CPHFMethod',
    'get_method',
]


"""
Utility functions for NLO calculations.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from utils.molecular import (
    smiles_to_geometry,
    canonicalize_smiles, 
    get_molecular_descriptors
)

__all__ = [
    # Molecular utilities
    'canonicalize_smiles',
    'smiles_to_geometry',
    'get_molecular_descriptors',
    'check_donor_groups',
    'check_acceptor_groups',
    'estimate_conjugation_length',
    # Analysis utilities
    'analyze_results',
    'compare_methods',
    'run_ranking_analysis',
    'plot_correlation',
]