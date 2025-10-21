# quantum_chemistry/calculators/__init__.py
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

try:
    from calculators.semiempirical import SemiEmpiricalCalculator
    calculators['semiempirical'] = SemiEmpiricalCalculator
except ImportError:
    pass

def get_calculator(name: str):
    """
    Get calculator class by name.
    
    Args:
        name: Calculator name ('dft', 'xtb', 'cc', 'semiempirical')
        
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
            raise ValueError("No calculators available. Install pyscf, tblite, or mopac.")
    return calculators[name.lower()]

__all__ = ['get_calculator'] + list(calculators.keys())