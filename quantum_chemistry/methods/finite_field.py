#!/usr/bin/env python3
"""
Basic finite field method for property estimation using DFTCalculator.
Fixed to use dipole_vector instead of scalar dipole.
"""
import numpy as np
from core.base import HyperpolarizabilityMethod, CalculationResult, Calculator

class FiniteFieldMethod(HyperpolarizabilityMethod):
    """
    Calculate first hyperpolarizability (β) using a basic finite field method.
    
    Uses 7 SCF calculations (zero field + ±x, ±y, ±z) to compute diagonal
    hyperpolarizability elements.
    """
    def __init__(self, calculator: Calculator, verbose: bool = False, field_strength: float = 0.001, **kwargs):
        super().__init__(calculator, verbose=verbose, **kwargs)
        self.field_strength = field_strength

    def calculate(self, atomic_numbers: np.ndarray, positions: np.ndarray,
                  charge: int = 0, spin: int = 1, **kwargs) -> CalculationResult:
        result = CalculationResult()
        try:
            h = kwargs.get('field_strength', self.field_strength)
            
            # Define field configurations
            field_directions = [
                [0.0, 0.0, 0.0],  # No field
                [h, 0.0, 0.0],    # +x
                [-h, 0.0, 0.0],   # -x
                [0.0, h, 0.0],    # +y
                [0.0, -h, 0.0],   # -y
                [0.0, 0.0, h],    # +z
                [0.0, 0.0, -h],   # -z
            ]
            
            # Collect dipole VECTORS (not magnitudes)
            dipoles = []
            energies = []
            
            for i, field in enumerate(field_directions):
                if self.verbose and i == 0:
                    print(f"  Calculating fields (0/{len(field_directions)})...", end='\r')
                
                calc_result = self.calculator.single_point(
                    atomic_numbers, positions, charge, spin, 
                    electric_field=np.array(field)
                )
                
                # FIXED: Use dipole_vector (3D array), not dipole (scalar)
                dipole_vec = calc_result.get('dipole_vector')
                
                if dipole_vec is None:
                    raise ValueError(f"Calculator returned None for dipole_vector at field {i}")
                
                dipoles.append(dipole_vec)
                energies.append(calc_result['energy'])
                
                if self.verbose:
                    print(f"  Calculating fields ({i+1}/{len(field_directions)})...", end='\r')
            
            if self.verbose:
                print()  # New line after progress
            
            # Convert to numpy array for indexing
            dipoles = np.array(dipoles)  # Shape: (7, 3)
            energies = np.array(energies)
            
            # Calculate diagonal beta elements using finite differences
            # β_iii = -d²μ_i/dE_i² = -(μ_i(+E) - 2μ_i(0) + μ_i(-E)) / h²
            
            beta_xxx = -(dipoles[1, 0] - 2*dipoles[0, 0] + dipoles[2, 0]) / (h**2)
            beta_yyy = -(dipoles[3, 1] - 2*dipoles[0, 1] + dipoles[4, 1]) / (h**2)
            beta_zzz = -(dipoles[5, 2] - 2*dipoles[0, 2] + dipoles[6, 2]) / (h**2)
            
            # Calculate beta vector magnitude
            beta_vec = np.sqrt(beta_xxx**2 + beta_yyy**2 + beta_zzz**2)
            
            # Store results
            result.beta_xxx = beta_xxx
            result.beta_yyy = beta_yyy
            result.beta_zzz = beta_zzz
            result.beta_vec = beta_vec
            result.beta_mean = (beta_xxx + beta_yyy + beta_zzz) / 3.0
            
            # Ground state properties (from zero field)
            result.dipole_vector = dipoles[0]
            result.dipole_moment = np.linalg.norm(dipoles[0])
            result.total_energy = energies[0]
            
            # Calculate polarizability as bonus (first derivative)
            alpha_xx = (dipoles[1, 0] - dipoles[2, 0]) / (2*h)
            alpha_yy = (dipoles[3, 1] - dipoles[4, 1]) / (2*h)
            alpha_zz = (dipoles[5, 2] - dipoles[6, 2]) / (2*h)
            result.alpha_mean = (alpha_xx + alpha_yy + alpha_zz) / 3.0
            
            # HOMO-LUMO gap
            try:
                calc_result_0 = self.calculator.single_point(
                    atomic_numbers, positions, charge, spin
                )
                homo, lumo = self.calculator.get_orbital_info(calc_result_0)
                result.homo_lumo_gap = (lumo - homo) * 27.211  # Convert to eV
            except:
                result.homo_lumo_gap = None
            
            # Skip gamma for speed (would need 15 more calculations)
            result.gamma = None
            
            if self.verbose:
                print(f"  Finite field β_vec: {beta_vec:.3e} a.u.")
                print(f"  β_xxx={beta_xxx:.3e}, β_yyy={beta_yyy:.3e}, β_zzz={beta_zzz:.3e}")
                if result.alpha_mean:
                    print(f"  α_mean: {result.alpha_mean:.3e} a.u.")
                    
        except Exception as e:
            result.error = f"Finite field calculation failed: {str(e)}"
            if self.verbose:
                print(f"  Error: {result.error}")
                import traceback
                traceback.print_exc()
        
        return result

    @property
    def name(self) -> str:
        return f"finite_field(h={self.field_strength})"