#!/usr/bin/env python3
"""
This module implements a comprehensive finite field approach for calculating second-order nonlinear optical properties
(hyperpolarizability) by computing the full β tensor using numerical differentiation of dipole moments under
various electric field configurations.
"""

import numpy as np
import itertools
from core.base import HyperpolarizabilityMethod, CalculationResult, Calculator


class FullTensorMethod(HyperpolarizabilityMethod):
    """
    Full tensor finite field method for hyperpolarizability calculation.
    
    This class implements a complete finite field approach to calculate the second-order hyperpolarizability
    tensor β_ijk by applying electric fields in various directions and computing numerical derivatives
    of the dipole moment. The method calculates all 27 tensor elements using finite difference formulas.
    
    The hyperpolarizability tensor β_ijk relates the second-order induced dipole moment to applied
    electric fields: μ_i^(2) = β_ijk * E_j * E_k
    
    Attributes:
        field_strength (float): Magnitude of electric field perturbations used in finite differences
        verbose (bool): Flag for detailed output during calculations
    """
    
    def __init__(self, calculator: Calculator, field_strength: float = 0.001, verbose: bool = False, **kwargs):
        """
        Initialize the full tensor hyperpolarizability method.
        
        Args:
            calculator (Calculator): Quantum chemistry calculator instance for single-point calculations
            field_strength (float, optional): Electric field strength for finite differences. Defaults to 0.001 a.u.
            verbose (bool, optional): Enable verbose output. Defaults to False.
            **kwargs: Additional keyword arguments passed to parent class
        """
        super().__init__(calculator, verbose=verbose, **kwargs)
        self.field_strength = field_strength

    def calculate(self, atomic_numbers: np.ndarray, positions: np.ndarray,
                  charge: int = 0, spin: int = 1, **kwargs) -> CalculationResult:
        """
        Calculate the complete hyperpolarizability tensor using finite field method.
        
        This method computes the second-order hyperpolarizability tensor by:
        1. Performing single-point calculations under various electric field configurations
        2. Extracting dipole moments from each calculation
        3. Using finite difference formulas to compute tensor elements
        4. Computing scalar quantities (β_vec, β_mean) from the full tensor
        
        The finite difference formulas used are:
        - Diagonal elements: β_iii = -(μ_i(+E_i) - 2μ_i(0) + μ_i(-E_i)) / h²
        - Mixed elements: β_iij = -(μ_i(+E_i,+E_j) - μ_i(+E_i,-E_j) - μ_i(-E_i,+E_j) + μ_i(-E_i,-E_j)) / (4h²)
        
        Args:
            atomic_numbers (np.ndarray): Atomic numbers for each atom
            positions (np.ndarray): Cartesian coordinates of atoms in atomic units
            charge (int, optional): Total molecular charge. Defaults to 0.
            spin (int, optional): Spin multiplicity. Defaults to 1.
            **kwargs: Additional keyword arguments, may include 'field_strength' override
            
        Returns:
            CalculationResult: Object containing calculated properties including:
                - beta_tensor: Full 3x3x3 hyperpolarizability tensor
                - beta_vec: Vector part of hyperpolarizability (magnitude)
                - beta_mean: Mean hyperpolarizability (rotationally averaged)
                - beta_xxx, beta_yyy, beta_zzz: Diagonal tensor elements
                - dipole_moment: Ground state dipole moment magnitude
                - dipole_vector: Ground state dipole moment vector
                - homo_lumo_gap: HOMO-LUMO gap in eV
                - total_energy: Total energy in a.u.
                - alpha_mean: Mean polarizability in a.u.
                - transition_dipole: Transition dipole moment
                - oscillator_strength: Oscillator strength
                - gamma: Second hyperpolarizability
        """
        result = CalculationResult()
        
        try:
            # Get field strength from kwargs or use instance default
            h = kwargs.get('field_strength', self.field_strength)
            
            # Create mapping for coordinate indices to axis labels
            idx_map = {0: 'x', 1: 'y', 2: 'z'}
            
            # Define all electric field configurations needed for finite differences
            # This includes zero field, single-axis fields, and two-axis combinations
            field_configs = {
                # Zero field reference
                '000': [0.0, 0.0, 0.0],
                
                # Single-axis positive and negative fields for diagonal elements
                '+x': [h, 0.0, 0.0], '-x': [-h, 0.0, 0.0],
                '+y': [0.0, h, 0.0], '-y': [0.0, -h, 0.0],
                '+z': [0.0, 0.0, h], '-z': [0.0, 0.0, -h],
                
                # Two-axis field combinations for off-diagonal elements
                '+x+y': [h, h, 0.0], '+x-y': [h, -h, 0.0], '-x+y': [-h, h, 0.0], '-x-y': [-h, -h, 0.0],
                '+x+z': [h, 0.0, h], '+x-z': [h, 0.0, -h], '-x+z': [-h, 0.0, h], '-x-z': [-h, 0.0, -h],
                '+y+z': [0.0, h, h], '+y-z': [0.0, h, -h], '-y+z': [0.0, -h, h], '-y-z': [0.0, -h, -h],
                
                # Additional permutations required for complete tensor calculation
                '+y-x': [0.0, h, -h], '-y+x': [0.0, -h, h], '+y+x': [0.0, h, h], '-y-x': [0.0, -h, -h],
                '+z-y': [h, 0.0, -h], '-z-y': [-h, 0.0, -h], '+z-x': [h, -h, 0.0], '-z-x': [-h, -h, 0.0],
                '-y-x': [0.0, -h, -h], '+z-x': [h, -h, 0.0], '-z+y': [-h, 0.0, h], '+y+x': [0.0, h, h],
                '+z+y': [h, h, 0.0], '+z+x': [h, h, 0.0], '-z+x': [-h, h, 0.0], '+z-y': [h, 0.0, -h],
            }
            
            # Perform single-point calculations for each field configuration
            # Store dipole moment vectors and energies for each configuration
            dipoles = {}
            energies = {}
            calc_results = {}
            
            for config_name, field in field_configs.items():
                calc_result = self.calculator.single_point(
                    atomic_numbers, positions, charge, spin, electric_field=np.array(field)
                )
                dipoles[config_name] = calc_result['dipole_vector']
                energies[config_name] = calc_result['energy']
                calc_results[config_name] = calc_result

            # Defensive programming: verify all required field configurations are present
            # Build set of required keys based on tensor symmetry requirements
            required_keys = set()
            
            # Single-axis fields needed for diagonal elements
            for i in range(3):
                dir_i = idx_map[i]
                required_keys.update([f'+{dir_i}', f'-{dir_i}'])
            
            # Two-axis combinations needed for off-diagonal elements
            for i in range(3):
                for j in range(3):
                    if i != j:
                        dir_i = idx_map[i]
                        dir_j = idx_map[j]
                        required_keys.update([
                            f'+{dir_i}+{dir_j}', f'+{dir_i}-{dir_j}',
                            f'-{dir_i}+{dir_j}', f'-{dir_i}-{dir_j}'
                        ])
            
            # Special combinations for xyz mixed derivatives
            required_keys.update(['+y+z', '+y-z', '-y+z', '-y-z', '000'])
            
            # Check for missing configurations and raise error if any are absent
            missing = [k for k in required_keys if k not in dipoles]
            if missing:
                raise KeyError(f"Missing dipole keys: {missing}. This may indicate a bug in field_configs or calculator output.")
            
            # Initialize the 3×3×3 hyperpolarizability tensor
            beta_tensor = np.zeros((3, 3, 3))
            
            # Calculate diagonal elements β_iii using second-order finite differences
            # Formula: β_iii = -(μ_i(+E_i) - 2μ_i(0) + μ_i(-E_i)) / h²
            for i in range(3):
                dir_i = idx_map[i]
                beta_tensor[i, i, i] = -(
                    dipoles[f'+{dir_i}'][i] - 2*dipoles['000'][i] + dipoles[f'-{dir_i}'][i]
                ) / (h**2)
            
            # Calculate off-diagonal elements β_iij using mixed second derivatives
            # Formula: β_iij = -(μ_i(+E_i,+E_j) - μ_i(+E_i,-E_j) - μ_i(-E_i,+E_j) + μ_i(-E_i,-E_j)) / (4h²)
            # Due to tensor symmetry, β_iij = β_iji = β_jii
            for i in range(3):
                for j in range(3):
                    if i != j:
                        dir_i = idx_map[i]
                        dir_j = idx_map[j]
                        key_pp = f'+{dir_i}+{dir_j}'
                        key_pm = f'+{dir_i}-{dir_j}'
                        key_mp = f'-{dir_i}+{dir_j}'
                        key_mm = f'-{dir_i}-{dir_j}'
                        
                        # Compute mixed derivative
                        beta_iij = -(
                            dipoles[key_pp][i] - dipoles[key_pm][i] -
                            dipoles[key_mp][i] + dipoles[key_mm][i]
                        ) / (4 * h**2)
                        
                        # Assign to all symmetric positions in tensor
                        beta_tensor[i, i, j] = beta_iij
                        beta_tensor[i, j, i] = beta_iij
                        beta_tensor[j, i, i] = beta_iij
            
            # Calculate β_xyz element using finite differences in y and z directions
            # This represents the fully off-diagonal (all different indices) tensor element
            beta_xyz = -(
                dipoles['+y+z'][0] - dipoles['+y-z'][0] -
                dipoles['-y+z'][0] + dipoles['-y-z'][0]
            ) / (4 * h**2)
            
            # Assign β_xyz to all permutations of (0,1,2) indices due to tensor symmetry
            for perm in itertools.permutations([0, 1, 2]):
                beta_tensor[perm] = beta_xyz
            
            # Store the complete tensor in results
            result.beta_tensor = beta_tensor
            
            # Extract and store diagonal tensor elements for easy access
            result.beta_xxx = beta_tensor[0, 0, 0]
            result.beta_yyy = beta_tensor[1, 1, 1]
            result.beta_zzz = beta_tensor[2, 2, 2]
            
            # Calculate vector part of hyperpolarizability
            # β_i = (β_iii + β_ijj + β_ikk) / 3 for each Cartesian direction
            beta_x = (beta_tensor[0, 0, 0] + beta_tensor[0, 1, 1] + beta_tensor[0, 2, 2]) / 3.0
            beta_y = (beta_tensor[1, 1, 1] + beta_tensor[1, 0, 0] + beta_tensor[1, 2, 2]) / 3.0
            beta_z = (beta_tensor[2, 2, 2] + beta_tensor[2, 0, 0] + beta_tensor[2, 1, 1]) / 3.0
            
            # Vector magnitude of hyperpolarizability
            result.beta_vec = np.sqrt(beta_x**2 + beta_y**2 + beta_z**2)
            
            # Calculate mean (rotationally averaged) hyperpolarizability
            # Formula: β_mean = (1/3)[Σ_i β_iii + 2 Σ_{i≠j} β_ijj]
            beta_mean = 0
            for i in range(3):
                # Diagonal contributions
                beta_mean += beta_tensor[i, i, i]
                for j in range(3):
                    if i != j:
                        # Off-diagonal contributions (factor of 2)
                        beta_mean += 2 * beta_tensor[i, j, j]
            result.beta_mean = beta_mean / 3.0
            
            # Extract ground state properties from zero-field calculation
            zero_field_result = calc_results['000']
            result.dipole_moment = np.linalg.norm(dipoles['000'])  # Magnitude
            result.dipole_vector = dipoles['000']                   # Vector
            result.total_energy = energies['000']                   # Total energy
            
            # Calculate HOMO-LUMO gap from zero-field calculation
            homo, lumo = self.calculator.get_orbital_info(zero_field_result)
            result.homo_lumo_gap = (lumo - homo) * 27.211  # Convert from a.u. to eV
            
            # Calculate polarizability (alpha) using the calculator's method
            try:
                polar_result = self.calculator.calculate_polarizability(atomic_numbers, positions, charge, spin)
                
                # Handle both dict and direct float returns
                if isinstance(polar_result, dict):
                    result.alpha_mean = polar_result.get('alpha_mean')
                elif isinstance(polar_result, (int, float, np.floating)):
                    result.alpha_mean = float(polar_result)
                else:
                    result.alpha_mean = None
            except Exception as e:
                if self.verbose:
                    print(f"  Warning: Could not calculate polarizability: {e}")
                result.alpha_mean = None

            # Calculate transition dipole and oscillator strength if calculator supports it
            try:
                if hasattr(self.calculator, 'calculate_transition_dipole'):
                    trans_result = self.calculator.calculate_transition_dipole(
                        atomic_numbers, positions, charge, spin, n_states=1
                    )
                    if trans_result and 'transition_dipole_magnitudes' in trans_result:
                        result.transition_dipole = trans_result['transition_dipole_magnitudes'][0] if trans_result['transition_dipole_magnitudes'] else None
                    if trans_result and 'oscillator_strengths' in trans_result:
                        result.oscillator_strength = trans_result['oscillator_strengths'][0] if trans_result['oscillator_strengths'] else None
            except Exception as e:
                if self.verbose:
                    print(f"  Warning: Could not calculate transition dipole: {e}")
                result.transition_dipole = None
                result.oscillator_strength = None
            
            # Calculate second hyperpolarizability (gamma) using finite field
            try:
                result.gamma = self._calculate_gamma(atomic_numbers, positions, charge, spin, h, energies)
            except Exception as e:
                if self.verbose:
                    print(f"  Warning: Could not calculate gamma: {e}")
            
            # Optional verbose output
            if self.verbose:
                print(f"  Full tensor β_vec: {result.beta_vec:.3e} a.u.")
                if result.alpha_mean:
                    print(f"  Alpha mean: {result.alpha_mean:.3e} a.u.")
                if result.gamma:
                    print(f"  Gamma mean: {result.gamma:.3e} a.u.")
                
        except Exception as e:
            # Handle any errors that occur during calculation
            result.error = f"Full tensor calculation failed: {str(e)}"
            if self.verbose:
                print(f"  Error: {result.error}")
        
        return result

    def _calculate_gamma(self, atomic_numbers: np.ndarray, positions: np.ndarray,
                     charge: int, spin: int, h: float, energies: dict) -> float:
        """
        Calculate mean second hyperpolarizability (gamma) using finite field.
        
        Args:
            atomic_numbers: Array of atomic numbers
            positions: Array of atomic positions in Angstrom
            charge: Molecular charge
            spin: Spin multiplicity (2S+1)
            h: Field strength used
            energies: Dictionary of energies from field calculations
            
        Returns:
            Mean gamma in a.u.
        """
        try:
            # For gamma, we need additional field points at ±2h
            # We'll calculate the diagonal components γ_iiii
            gamma_components = []
            
            for i in range(3):
                dir_i = ['x', 'y', 'z'][i]
                
                # We need energies at 0, ±h, ±2h
                E_0 = energies['000']
                E_p1 = energies[f'+{dir_i}']
                E_m1 = energies[f'-{dir_i}']
                
                # Calculate ±2h field points
                field_p2 = np.zeros(3)
                field_p2[i] = 2 * h
                result_p2 = self.calculator.single_point(
                    atomic_numbers, positions, charge, spin, electric_field=field_p2
                )
                E_p2 = result_p2['energy']
                
                field_m2 = np.zeros(3)
                field_m2[i] = -2 * h
                result_m2 = self.calculator.single_point(
                    atomic_numbers, positions, charge, spin, electric_field=field_m2
                )
                E_m2 = result_m2['energy']
                
                # Fourth derivative using finite difference
                # f''''(0) ≈ [f(-2h) - 4f(-h) + 6f(0) - 4f(h) + f(2h)] / h^4
                # gamma = - d^4E / dF^4 (per standard energy expansion convention)
                gamma_iiii = - (E_m2 - 4*E_m1 + 6*E_0 - 4*E_p1 + E_p2) / (h**4)
                gamma_components.append(gamma_iiii)
            
            # Mean gamma (average of diagonal components)
            gamma_mean = np.mean(gamma_components)
            
            return gamma_mean
        
        except Exception as e:
            if self.verbose:
                print(f"  Error calculating gamma: {e}")
            return None

    @property
    def name(self) -> str:
        """
        Return descriptive name for this method including field strength parameter.
        
        Returns:
            str: Method name with field strength specification
        """
        return f"full_tensor(h={self.field_strength})"