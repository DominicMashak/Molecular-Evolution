#!/usr/bin/env python3
"""
Coupled-Perturbed Hartree-Fock (CPHF) method for hyperpolarizability calculation.

This method calculates hyperpolarizability analytically using response theory,
which is much faster than finite field methods. It solves the CPHF equations
directly rather than using numerical differentiation.
"""

import numpy as np
from core.base import HyperpolarizabilityMethod, CalculationResult, Calculator

PYSCF_AVAILABLE = False
polarizability = None

try:
    import pyscf
    PYSCF_AVAILABLE = True
    
    try:
        from pyscf.prop import polarizability
    except (ImportError, AttributeError):
        try:
            import pyscf.prop.polarizability as polarizability
        except ImportError:
            try:
                from pyscf.properties import polarizability
            except ImportError:
                PYSCF_AVAILABLE = False
                polarizability = None

except ImportError:
    PYSCF_AVAILABLE = False


class CPHFMethod(HyperpolarizabilityMethod):
    """
    Analytical CPHF method for hyperpolarizability calculation.
    
    This method uses coupled-perturbed Hartree-Fock theory to calculate
    hyperpolarizability analytically. Instead of running 27+ SCF calculations
    with different electric fields, it solves the CPHF equations once to get
    the response properties.
    
    Identical to converged finite field for HF calculations
    
    Requirements:
    - PySCF with polarizability module
    - Works best with HF-based calculators (DFT support varies by functional)
    
    Usage:
        calculator = DFTCalculator(functional='HF', basis='sto-3g')
        method = CPHFMethod(calculator, verbose=True)
        result = method.calculate(atomic_numbers, positions, charge, spin)
    """
    
    def __init__(self, calculator: Calculator, verbose: bool = False, **kwargs):
        """
        Initialize CPHF method.
        
        Args:
            calculator: Quantum chemistry calculator (must support single_point)
            verbose: Enable detailed output during calculation
            **kwargs: Additional options (currently unused, for compatibility)
        """
        super().__init__(calculator, verbose=verbose, **kwargs)
        
        if not PYSCF_AVAILABLE:
            if verbose:
                print("  Warning: PySCF not available - will use fallback finite field")
        elif polarizability is None:
            if verbose:
                print("  Warning: PySCF polarizability module not found - will use fallback")
        
        self.config = kwargs
        self.use_cphf = PYSCF_AVAILABLE and polarizability is not None
    
    def calculate(self, atomic_numbers: np.ndarray, positions: np.ndarray,
                  charge: int = 0, spin: int = 1, **kwargs) -> CalculationResult:
        """
        Calculate hyperpolarizability using analytical CPHF method.
        
        This performs:
        1. Single SCF calculation (no electric field)
        2. Analytical calculation of polarizability (alpha) via CPHF
        3. Analytical calculation of hyperpolarizability (beta) via CPHF
        4. Extraction of orbital energies for HOMO-LUMO gap
        
        If CPHF is not available, falls back to finite field method.
        
        Args:
            atomic_numbers: Array of atomic numbers (shape: [n_atoms])
            positions: Atomic coordinates in Angstrom (shape: [n_atoms, 3])
            charge: Total molecular charge
            spin: Spin multiplicity (2S+1)
            **kwargs: Additional options (unused, for compatibility)
            
        Returns:
            CalculationResult object containing:
                - beta_tensor: Full 3x3x3 hyperpolarizability tensor
                - beta_vec: Vector magnitude of hyperpolarizability
                - beta_mean: Mean (rotationally averaged) hyperpolarizability
                - beta_xxx, beta_yyy, beta_zzz: Diagonal tensor elements
                - alpha_mean: Mean polarizability
                - dipole_moment: Ground state dipole moment magnitude
                - dipole_vector: Ground state dipole moment vector
                - homo_lumo_gap: HOMO-LUMO gap in eV
                - total_energy: Total electronic energy in a.u.
                - gamma: Second hyperpolarizability (not calculated for speed)
        """
        # If CPHF not available, use fallback immediately
        if not self.use_cphf:
            if self.verbose:
                print("  CPHF not available, using finite field method...")
            return self._fallback_finite_field(atomic_numbers, positions, charge, spin)
        
        result = CalculationResult()
        
        try:
            # Step 1: Perform ground state SCF calculation
            if self.verbose:
                print("  Step 1/4: Running ground state SCF...")
            
            calc_result = self.calculator.single_point(
                atomic_numbers, positions, charge, spin
            )
            
            # Check convergence
            if not calc_result.get('converged', False):
                result.error = "SCF did not converge"
                if self.verbose:
                    print(f"  Error: {result.error}")
                return result
            
            mf = calc_result['mf']
            mol = calc_result['mol']
            
            if self.verbose:
                print(f"  SCF converged: E = {calc_result['energy']:.6f} a.u.")
            
            # Step 2: Calculate polarizability (alpha) analytically
            if self.verbose:
                print("  Step 2/4: Calculating polarizability (CPHF)...")
            
            try:
                # PySCF's analytical polarizability calculation
                # Returns 3x3 tensor: alpha[i,j] = dμ_i/dE_j
                alpha_tensor = polarizability.polarizability(mf)
                
                # Mean polarizability (trace / 3)
                result.alpha_mean = np.trace(alpha_tensor) / 3.0
                
                if self.verbose:
                    print(f"  α_mean = {result.alpha_mean:.3e} a.u.")
                    print(f"  α tensor diagonal: [{alpha_tensor[0,0]:.2e}, "
                          f"{alpha_tensor[1,1]:.2e}, {alpha_tensor[2,2]:.2e}]")
            
            except Exception as e:
                if self.verbose:
                    print(f"  Warning: Polarizability calculation failed: {e}")
                result.alpha_mean = None
            
            # Step 3: Calculate hyperpolarizability (beta) analytically
            if self.verbose:
                print("  Step 3/4: Calculating hyperpolarizability (CPHF)...")
            
            try:
                # Check if hyper_polarizability function exists in this PySCF version
                if not hasattr(polarizability, 'hyper_polarizability'):
                    raise AttributeError(
                        "PySCF version does not have hyper_polarizability function. "
                        "This requires PySCF >= 2.1.0"
                    )
                
                # Calculate full beta tensor analytically
                # beta[i,j,k] = d²μ_i/dE_j dE_k
                # with_cphf=True ensures full CPHF solve (not finite difference)
                beta_tensor = polarizability.hyper_polarizability(mf, with_cphf=True)
                
                # Store full tensor
                result.beta_tensor = beta_tensor
                
                # Extract diagonal elements
                result.beta_xxx = beta_tensor[0, 0, 0]
                result.beta_yyy = beta_tensor[1, 1, 1]
                result.beta_zzz = beta_tensor[2, 2, 2]
                
                if self.verbose:
                    print(f"  β_xxx = {result.beta_xxx:.3e} a.u.")
                    print(f"  β_yyy = {result.beta_yyy:.3e} a.u.")
                    print(f"  β_zzz = {result.beta_zzz:.3e} a.u.")
                
                # Calculate beta vector (same formula as full_tensor.py)
                # β_i = (β_iii + β_ijj + β_ikk) / 3
                beta_x = (beta_tensor[0, 0, 0] + beta_tensor[0, 1, 1] + beta_tensor[0, 2, 2]) / 3.0
                beta_y = (beta_tensor[1, 1, 1] + beta_tensor[1, 0, 0] + beta_tensor[1, 2, 2]) / 3.0
                beta_z = (beta_tensor[2, 2, 2] + beta_tensor[2, 0, 0] + beta_tensor[2, 1, 1]) / 3.0
                
                result.beta_vec = np.sqrt(beta_x**2 + beta_y**2 + beta_z**2)
                
                # Calculate mean hyperpolarizability (same formula as full_tensor.py)
                # β_mean = (1/3)[Σ_i β_iii + 2 Σ_{i≠j} β_ijj]
                beta_mean = 0.0
                for i in range(3):
                    # Diagonal contributions
                    beta_mean += beta_tensor[i, i, i]
                    for j in range(3):
                        if i != j:
                            # Off-diagonal contributions (factor of 2)
                            beta_mean += 2 * beta_tensor[i, j, j]
                
                result.beta_mean = beta_mean / 3.0
                
                if self.verbose:
                    print(f"  β_vec = {result.beta_vec:.3e} a.u.")
                    print(f"  β_mean = {result.beta_mean:.3e} a.u.")
            
            except (AttributeError, NotImplementedError) as e:
                # PySCF version doesn't have hyper_polarizability or it's not implemented
                if self.verbose:
                    print(f"  Warning: {e}")
                    print("  Falling back to simplified finite field method...")
                
                # Fallback to 7-point finite field (like finite_field.py)
                return self._fallback_finite_field(atomic_numbers, positions, charge, spin)
            
            except Exception as e:
                if self.verbose:
                    print(f"  Warning: CPHF hyperpolarizability failed: {e}")
                    print("  Falling back to simplified finite field method...")
                
                # Fallback to finite field
                return self._fallback_finite_field(atomic_numbers, positions, charge, spin)
            
            # Step 4: Extract other molecular properties
            if self.verbose:
                print("  Step 4/4: Extracting molecular properties...")
            
            # Dipole moment
            result.dipole_vector = calc_result.get('dipole_vector', np.zeros(3))
            result.dipole_moment = calc_result.get('dipole', 0.0)
            
            # Total energy
            result.total_energy = calc_result['energy']
            
            # HOMO-LUMO gap
            try:
                homo, lumo = self.calculator.get_orbital_info(calc_result)
                result.homo_lumo_gap = (lumo - homo) * 27.211  # Convert Hartree to eV
                
                if self.verbose:
                    print(f"  HOMO-LUMO gap = {result.homo_lumo_gap:.3f} eV")
            except:
                result.homo_lumo_gap = None
            
            # Gamma (second hyperpolarizability) - skip for speed
            # Would require solving 4th-order CPHF equations
            result.gamma = None
            
            if self.verbose:
                print("  CPHF calculation completed successfully!")
        
        except Exception as e:
            result.error = f"CPHF calculation failed: {str(e)}"
            if self.verbose:
                print(f"  Error: {result.error}")
                import traceback
                traceback.print_exc()
        
        return result
    
    def _fallback_finite_field(self, atomic_numbers: np.ndarray, positions: np.ndarray,
                               charge: int, spin: int) -> CalculationResult:
        """
        Fallback method using simplified finite field (7 calculations).
        
        Used when CPHF is not available or fails. This is the same approach
        as finite_field.py - calculates only diagonal beta elements.
        
        Args:
            atomic_numbers: Array of atomic numbers
            positions: Atomic coordinates in Angstrom
            charge: Total molecular charge
            spin: Spin multiplicity
            
        Returns:
            CalculationResult with diagonal beta elements
        """
        result = CalculationResult()
        h = 0.001  # Field strength in a.u.
        
        if self.verbose:
            print("  Using 7-point finite field method...")
        
        try:
            # Zero field calculation
            calc_0 = self.calculator.single_point(atomic_numbers, positions, charge, spin)
            mu_0 = calc_0.get('dipole_vector', np.zeros(3))
            E_0 = calc_0['energy']
            
            beta_diag = []
            alpha_diag = []
            
            # Calculate along each axis
            for i in range(3):
                # Positive field
                field_pos = np.zeros(3)
                field_pos[i] = h
                calc_pos = self.calculator.single_point(
                    atomic_numbers, positions, charge, spin,
                    electric_field=field_pos
                )
                mu_pos = calc_pos.get('dipole_vector', np.zeros(3))
                
                # Negative field
                field_neg = np.zeros(3)
                field_neg[i] = -h
                calc_neg = self.calculator.single_point(
                    atomic_numbers, positions, charge, spin,
                    electric_field=field_neg
                )
                mu_neg = calc_neg.get('dipole_vector', np.zeros(3))
                
                # Second derivative: beta_iii
                beta_ii = -(mu_pos[i] - 2*mu_0[i] + mu_neg[i]) / (h**2)
                beta_diag.append(beta_ii)
                
                # First derivative: alpha_ii
                alpha_ii = (mu_pos[i] - mu_neg[i]) / (2*h)
                alpha_diag.append(alpha_ii)
            
            # Store results
            result.beta_xxx = beta_diag[0]
            result.beta_yyy = beta_diag[1]
            result.beta_zzz = beta_diag[2]
            result.beta_vec = np.sqrt(sum(b**2 for b in beta_diag))
            result.beta_mean = np.mean(beta_diag)
            
            result.alpha_mean = np.mean(alpha_diag)
            
            result.dipole_vector = mu_0
            result.dipole_moment = np.linalg.norm(mu_0)
            result.total_energy = E_0
            
            # HOMO-LUMO gap
            try:
                homo, lumo = self.calculator.get_orbital_info(calc_0)
                result.homo_lumo_gap = (lumo - homo) * 27.211
            except:
                result.homo_lumo_gap = None
            
            result.gamma = None
            
            if self.verbose:
                print(f"  Fallback β_vec = {result.beta_vec:.3e} a.u.")
        
        except Exception as e:
            result.error = f"Fallback finite field failed: {str(e)}"
            if self.verbose:
                print(f"  Error: {result.error}")
        
        return result
    
    @property
    def name(self) -> str:
        """Return method name for identification."""
        if self.use_cphf:
            return "cphf"
        else:
            return "cphf(fallback_ff)"