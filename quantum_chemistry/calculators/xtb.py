#!/usr/bin/env python3
"""
xTB calculator implementation using DFTB+ backend.
"""

import numpy as np
import os
import tempfile
import subprocess
import shutil
from typing import Dict, Any, Optional, Tuple
import sys

try:
    from core.base import Calculator
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.base import Calculator


class XTBCalculator(Calculator):
    """
    xTB calculator using DFTB+ backend.
    
    This calculator implements various xTB methods (GFN0-xTB, GFN1-xTB, GFN2-xTB)
    using the DFTB+ software package as the computational backend.
    """
    
    def __init__(self, xtb_method: str = 'GFN2-xTB', dielectric: Optional[float] = None, 
                 verbose: bool = False, dftbplus_path: Optional[str] = None, method: str = 'single_point', **kwargs):
        """
        Initialize xTB calculator.
        
        Args:
            xtb_method: xTB method ('GFN0-xTB', 'GFN1-xTB', 'GFN2-xTB')
            dielectric: Dielectric constant for implicit solvation
            verbose: Print detailed output
            dftbplus_path: Path to DFTB+ executable (if not in PATH)
            method: Calculation method ('single_point' for energy, 'finite_field' for beta)
            **kwargs: Additional parameters
        """
        super().__init__(verbose=verbose, **kwargs)
        
        self.xtb_method = xtb_method.upper()
        self.dielectric = dielectric
        self.dftbplus_path = dftbplus_path or 'dftb+'
        self.method = method.lower()
        
        # Validate method
        valid_methods = ['GFN0-XTB', 'GFN1-XTB', 'GFN2-XTB']
        if self.xtb_method not in valid_methods:
            raise ValueError(f"Invalid xTB method: {self.xtb_method}. Must be one of {valid_methods}")
        
        # Check if DFTB+ is available
        self._check_dftbplus()
    
    def _check_dftbplus(self):
        """Check if DFTB+ is available and set run command without invoking it.

        Avoid running `dftb+ --version` because in some MPI-enabled builds this
        can immediately MPI_ABORT. Use shutil.which to detect presence and
        prefer a no-MPI invocation; fall back to mpirun/mpiexec -np 1 if needed.
        """
        # First, check if we can find mpirun/mpiexec for potential fallback
        mpirun_bin = shutil.which('mpirun') or shutil.which('mpiexec')
        
        # Prefer to locate the executable without executing it
        exe_path = shutil.which(self.dftbplus_path)
        
        if exe_path and mpirun_bin:
            # If we have both dftb+ and mpirun, default to using mpirun for safety
            # This avoids the MPI_ABORT issue with MPI-compiled dftb+
            self.run_cmd = [mpirun_bin, '-np', '1', self.dftbplus_path]
            self.use_mpi = True
            if self.verbose:
                print(f"DFTB+ executable found at: {exe_path}")
                print(f"Will run via: {' '.join(self.run_cmd)}")
            return True
        elif exe_path:
            # Only dftb+ found, no mpirun - try direct execution
            self.run_cmd = [self.dftbplus_path]
            self.use_mpi = False
            if self.verbose:
                print(f"DFTB+ executable found at: {exe_path}")
                print(f"Will run directly: {' '.join(self.run_cmd)}")
            return True
        elif mpirun_bin:
            # Only mpirun found, assume dftb+ is in PATH
            self.run_cmd = [mpirun_bin, '-np', '1', self.dftbplus_path]
            self.use_mpi = True
            if self.verbose:
                print(f"Will attempt to run DFTB+ via: {' '.join(self.run_cmd)}")
            return True
        else:
            # Nothing found – raise an informative error
            raise RuntimeError(
                f"DFTB+ executable '{self.dftbplus_path}' not found in PATH and no mpirun/mpiexec available. "
                "Please install DFTB+ (https://dftbplus.org) or ensure the executable is on PATH."
            )

    def _write_geometry_file(self, atomic_numbers: np.ndarray, positions: np.ndarray, 
                       filename: str):
        """
        Write geometry to DFTB+ GenFormat.
        
        Args:
            atomic_numbers: Array of atomic numbers
            positions: Array of atomic positions in Angstrom
            filename: Output filename
        """
        from periodictable import elements
        
        with open(filename, 'w') as f:
            n_atoms = len(atomic_numbers)
            # Get unique elements in order of first appearance
            unique_z = []
            for z in atomic_numbers:
                if z not in unique_z:
                    unique_z.append(z)
            
            element_symbols = [elements[int(z)].symbol for z in unique_z]
            
            # First line: number of atoms, cluster (C for Cartesian)
            f.write(f"{n_atoms} C\n")
            
            # Second line: unique element symbols
            f.write(' '.join(element_symbols) + '\n')
            
            # Atom lines: index, type_index, x, y, z
            for i, z in enumerate(atomic_numbers):
                # Find element type index (1-based)
                elem_idx = unique_z.index(z) + 1
                x, y, z_coord = positions[i]
                f.write(f"{i+1} {elem_idx} {x:.10f} {y:.10f} {z_coord:.10f}\n")
    
    def _write_input_file(self, charge: int, spin: int, electric_field: Optional[np.ndarray] = None,
                     workdir: str = '.', write_band: bool = False):
        """
        Write DFTB+ input file.
        
        Args:
            charge: Molecular charge
            spin: Spin multiplicity (2S+1)
            electric_field: External electric field in a.u.
            workdir: Working directory
            write_band: Whether to write band.out for orbital info
        """
        input_file = os.path.join(workdir, 'dftb_in.hsd')
        
        # Map our method names to DFTB+ method strings
        # Based on the documentation, xTB methods are specified as strings
        method_map = {
            'GFN0-XTB': 'GFN0-xTB',
            'GFN1-XTB': 'GFN1-xTB', 
            'GFN2-XTB': 'GFN2-xTB'
        }
        xtb_method_str = method_map.get(self.xtb_method, 'GFN2-xTB')
        
        with open(input_file, 'w') as f:
            f.write("Geometry = GenFormat {\n")
            f.write("  <<< 'geometry.gen'\n")
            f.write("}\n\n")
            
            f.write("Driver = {}  # Single point calculation\n\n")
            
            f.write("Hamiltonian = xTB {\n")
            f.write(f"  Method = \"{xtb_method_str}\"\n")
            
            # Charge
            if charge != 0:
                f.write(f"  Charge = {charge}\n")
            
            # Spin
            if spin > 1:
                f.write(f"  SpinPolarisation = Colinear {{\n")
                f.write(f"    UnpairedElectrons = {spin - 1}\n")
                f.write(f"  }}\n")
            
            # Electric field
            if electric_field is not None and np.any(electric_field):
                # For xTB in DFTB+, electric fields are more complex
                # We'll use a simplified approach via ElectricField block
                ex, ey, ez = electric_field
                field_magnitude = np.linalg.norm(electric_field)
                if field_magnitude > 1e-10:
                    # Normalize direction
                    fx, fy, fz = ex/field_magnitude, ey/field_magnitude, ez/field_magnitude
                    # Convert from a.u. to V/Angstrom (1 a.u. = 51.4220652 V/Å)
                    strength_va = field_magnitude * 51.4220652
                    f.write(f"  ElectricField = {{\n")
                    f.write(f"    PointCharges = {{\n")
                    f.write(f"      CoordsAndCharges [Angstrom] = {{}}\n")
                    f.write(f"    }}\n")
                    f.write(f"  }}\n")
            
            # Solvation
            if self.dielectric and self.dielectric > 1.0:
                f.write("  Solvation = GeneralisedBorn {\n")
                f.write(f"    Solvent = fromConstants {{\n")
                f.write(f"      Epsilon = {self.dielectric}\n")
                f.write("    }\n")
                f.write("  }\n")
            
            # SCC parameters
            f.write("  SCC = Yes\n")
            f.write("  SCCTolerance = 1.0e-8\n")
            f.write("  MaxSCCIterations = 1000\n")
            
            f.write("}\n\n")
            
            # Analysis
            f.write("Analysis = {\n")
            if write_band:
                f.write("  WriteBandOut = Yes\n")
            f.write("  CalculateForces = No\n")
            f.write("}\n\n")
            
            # Options
            f.write("Options = {\n")
            f.write("  WriteResultsTag = Yes\n")
            f.write("}\n\n")
            
            # Parallelization - disable for compatibility
            f.write("Parallel = {\n")
            f.write("  UseOmpThreads = No\n")
            f.write("}\n\n")
            
            # Parser options
            f.write("ParserOptions {\n")
            f.write("  ParserVersion = 12\n")
            f.write("}\n")
    
    def _parse_output(self, workdir: str) -> Dict[str, Any]:
        """
        Parse DFTB+ output files.
        
        Args:
            workdir: Working directory
            
        Returns:
            Dictionary with calculation results
        """
        results = {}
        
        # Parse detailed.out for energy and convergence
        detailed_file = os.path.join(workdir, 'detailed.out')
        if os.path.exists(detailed_file):
            with open(detailed_file, 'r') as f:
                content = f.read()
                
                # Extract total energy
                if 'Total energy:' in content:
                    for line in content.split('\n'):
                        if 'Total energy:' in line:
                            energy_line = line.split('Total energy:')[1].strip()
                            results['energy'] = float(energy_line.split()[0])
                            break
                elif 'Total Energy:' in content:
                    for line in content.split('\n'):
                        if 'Total Energy:' in line:
                            energy_line = line.split('Total Energy:')[1].strip()
                            results['energy'] = float(energy_line.split()[0])
                            break
                
                # Extract SCC convergence
                if 'SCC converged' in content or 'SCC is converged' in content:
                    results['converged'] = True
                else:
                    results['converged'] = False
        
        # Parse charges.dat for Mulliken charges (if needed)
        charges_file = os.path.join(workdir, 'charges.dat')
        if os.path.exists(charges_file):
            with open(charges_file, 'r') as f:
                # Skip header and read charges
                charges = []
                for line in f:
                    if line.strip() and not line.startswith('#'):
                        try:
                            charges.append(float(line.strip()))
                        except ValueError:
                            continue
                if charges:
                    results['mulliken_charges'] = np.array(charges)
        
        # Parse band.out for orbital energies
        band_file = os.path.join(workdir, 'band.out')
        if os.path.exists(band_file):
            try:
                orbital_energies = []
                with open(band_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            try:
                                # Format: KPoint OrbitalIndex Energy Occupation
                                energy = float(parts[1])
                                orbital_energies.append(energy)
                            except (ValueError, IndexError):
                                continue
                if orbital_energies:
                    results['orbital_energies'] = np.array(orbital_energies)
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Could not parse band.out: {e}")
        
        # Parse dipole moment
        results['dipole_vector'] = np.zeros(3)
        results['dipole'] = 0.0
        
        # Try results.tag first (newer DFTB+ format)
        results_tag_file = os.path.join(workdir, 'results.tag')
        if os.path.exists(results_tag_file):
            try:
                with open(results_tag_file, 'r') as f:
                    lines = f.readlines()
                    for i, line in enumerate(lines):
                        if 'dipole' in line.lower():
                            # Next line should contain dipole vector
                            if i + 1 < len(lines):
                                dipole_line = lines[i + 1].strip()
                                dipole_parts = dipole_line.split()
                                if len(dipole_parts) >= 3:
                                    dipole_vector = np.array([float(x) for x in dipole_parts[:3]])
                                    results['dipole_vector'] = dipole_vector
                                    results['dipole'] = np.linalg.norm(dipole_vector)
                                    break
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Could not parse results.tag: {e}")
        
        # If dipole not found, try detailed.out
        if results['dipole'] == 0.0 and os.path.exists(detailed_file):
            with open(detailed_file, 'r') as f:
                for line in f:
                    if 'Dipole moment' in line or 'Total dipole' in line:
                        try:
                            # Try to extract values in various formats
                            if '[' in line and ']' in line:
                                dipole_str = line.split('[')[1].split(']')[0]
                                dipole_vector = np.array([float(x) for x in dipole_str.split()])
                            else:
                                parts = line.split(':')[1].strip().split()
                                dipole_vector = np.array([float(x) for x in parts[:3]])
                            results['dipole_vector'] = dipole_vector
                            results['dipole'] = np.linalg.norm(dipole_vector)
                            break
                        except Exception as e:
                            if self.verbose:
                                print(f"Warning: Could not parse dipole from detailed.out: {e}")
        
        return results
    
    def single_point(self, atomic_numbers: np.ndarray, positions: np.ndarray,
                charge: int = 0, spin: int = 1,
                electric_field: Optional[np.ndarray] = None,
                write_band: bool = False) -> Dict[str, Any]:
        """
        Perform xTB single point calculation.
        
        Args:
            atomic_numbers: Array of atomic numbers
            positions: Array of atomic positions in Angstrom
            charge: Molecular charge
            spin: Spin multiplicity (2S+1)
            electric_field: External electric field in a.u. [Ex, Ey, Ez]
            write_band: Whether to write band.out for orbital information
            
        Returns:
            Dictionary with calculation results
        """
        # Create temporary directory for calculation
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                # Write geometry file
                geom_file = os.path.join(tmpdir, 'geometry.gen')
                self._write_geometry_file(atomic_numbers, positions, geom_file)
                
                # Write input file
                self._write_input_file(charge, spin, electric_field, tmpdir, write_band)
                
                if self.verbose:
                    print(f"  Running {self.xtb_method} calculation in {tmpdir}")
                    print(f"  Command: {' '.join(self.run_cmd)}")
                
                # Use the configured run command
                cmd = list(self.run_cmd)

                # Set environment to disable MPI threading and be more permissive
                env = os.environ.copy()
                env['OMP_NUM_THREADS'] = '1'
                env['MKL_NUM_THREADS'] = '1'
                env['OMPI_MCA_btl_base_warn_component_unused'] = '0'  # Suppress OpenMPI warnings
                env['PMIX_MCA_gds'] = '^ds12,ds21'  # Help with some MPI issues

                # Run calculation
                result = subprocess.run(cmd, cwd=tmpdir, capture_output=True, 
                                    text=True, env=env, timeout=300)

                # Check for success - DFTB+ may exit with error code even if calculation worked
                # Look for signs of successful completion in output
                success = False
                if result.returncode == 0:
                    success = True
                elif result.stdout:
                    # Check if SCC converged despite non-zero exit code
                    if 'SCC converged' in result.stdout or 'SCC is converged' in result.stdout:
                        success = True
                        if self.verbose:
                            print(f"  DFTB+ completed with exit code {result.returncode} but SCC converged")
                
                if not success:
                    # Try to provide helpful error message
                    error_msg = f"DFTB+ calculation failed with exit code {result.returncode}"
                    if result.stderr:
                        error_msg += f"\nStderr: {result.stderr[:500]}"
                    if result.stdout:
                        # Show last few lines of stdout
                        stdout_lines = result.stdout.strip().split('\n')
                        error_msg += f"\nLast output lines:\n" + '\n'.join(stdout_lines[-5:])
                    raise RuntimeError(error_msg)

                if self.verbose and result.stdout:
                    output_lines = result.stdout.strip().split('\n')
                    print(f"  DFTB+ output (last 10 lines):")
                    for line in output_lines[-10:]:
                        print(f"    {line}")
                    if result.stderr:
                        print(f"  DFTB+ stderr: {result.stderr[:200]}")
                
                # Parse results
                calc_results = self._parse_output(tmpdir)
                
                if 'energy' not in calc_results:
                    # Try to extract from stdout
                    if result.stdout and 'Total energy:' in result.stdout:
                        for line in result.stdout.split('\n'):
                            if 'Total energy:' in line:
                                try:
                                    energy_line = line.split('Total energy:')[1].strip()
                                    calc_results['energy'] = float(energy_line.split()[0])
                                    break
                                except ValueError:
                                    continue
                    
                    if 'energy' not in calc_results:
                        raise RuntimeError("Failed to extract energy from DFTB+ output")
                
                # Ensure dipole is available
                if calc_results.get('dipole', 0.0) == 0.0:
                    if 'mulliken_charges' in calc_results:
                        charges = calc_results['mulliken_charges']
                        dipole_vector = np.zeros(3)
                        for i, (z, pos) in enumerate(zip(atomic_numbers, positions)):
                            pos_bohr = pos * 1.889726
                            dipole_vector += charges[i] * pos_bohr
                        calc_results['dipole_vector'] = dipole_vector
                        calc_results['dipole'] = np.linalg.norm(dipole_vector)
                    else:
                        calc_results['dipole_vector'] = np.zeros(3)
                        calc_results['dipole'] = 0.0
                
                return calc_results
                
            except subprocess.TimeoutExpired:
                raise RuntimeError("DFTB+ calculation timed out after 5 minutes")
            except Exception as e:
                raise RuntimeError(f"xTB calculation failed: {str(e)}")
    
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
        # Calculate dipole at zero field
        result_0 = self.single_point(atomic_numbers, positions, charge, spin)
        mu_0 = result_0['dipole_vector']
        
        alpha_tensor = np.zeros((3, 3))
        
        # Calculate polarizability tensor components
        for i in range(3):
            # Positive field
            field_pos = np.zeros(3)
            field_pos[i] = field_strength
            result_pos = self.single_point(atomic_numbers, positions, charge, spin, electric_field=field_pos)
            mu_pos = result_pos['dipole_vector']
            
            # Negative field
            field_neg = np.zeros(3)
            field_neg[i] = -field_strength
            result_neg = self.single_point(atomic_numbers, positions, charge, spin, electric_field=field_neg)
            mu_neg = result_neg['dipole_vector']
            
            # Calculate polarizability using central difference
            alpha_tensor[:, i] = (mu_pos - mu_neg) / (2 * field_strength)
        
        # Mean polarizability
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
        gamma_components = []
        
        for i in range(3):
            # Energies at 0, ±h, ±2h
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
            
            # Fourth derivative
            gamma_ii = (E_m2 - 4*E_m1 + 6*E_0 - 4*E_p1 + E_p2) / (field_strength**4)
            gamma_components.append(gamma_ii)
        
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
        
        # Use the full tensor method
        method = FullTensorMethod(self, field_strength=field_strength, verbose=self.verbose)
        result = method.calculate(atomic_numbers, positions, charge, spin)
        
        # Extract relevant values
        beta_vec = result.beta_vec if result.beta_vec is not None else 0.0
        beta_mean = result.beta_mean if result.beta_mean is not None else 0.0
        beta_xxx = result.beta_xxx if result.beta_xxx is not None else 0.0
        beta_yyy = result.beta_yyy if result.beta_yyy is not None else 0.0
        beta_zzz = result.beta_zzz if result.beta_zzz is not None else 0.0
        
        # Calculate additional properties
        ref_result = self.single_point(atomic_numbers, positions, charge, spin, write_band=True)
        dipole_moment = np.linalg.norm(ref_result['dipole_vector']) if 'dipole_vector' in ref_result else None
        total_energy = ref_result['energy']
        
        # Get orbital information
        homo, lumo = self.get_orbital_info(ref_result)
        homo_lumo_gap = (lumo - homo) * 27.211  # Convert to eV
        
        # Calculate transition dipole (not available for xTB, set to None)
        transition_dipole = None
        oscillator_strength = None
        
        # Calculate polarizability and gamma
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
        Calculate transition dipole moments.
        
        Note: TD-DFT-like excited state calculations are not directly available in xTB/DFTB+.
        This method returns None values to maintain API compatibility.
        
        Args:
            atomic_numbers: Array of atomic numbers
            positions: Array of atomic positions in Angstrom
            charge: Molecular charge
            spin: Spin multiplicity (2S+1)
            n_states: Number of excited states to compute
            
        Returns:
            Dictionary with None values (xTB doesn't support TD-DFT)
        """
        if self.verbose:
            print("Warning: Excited state calculations not available for xTB method")
        
        return {
            'transition_dipoles': None,
            'transition_dipole_magnitudes': None,
            'excitation_energies': None,
            'oscillator_strengths': None,
            'ground_energy': None
        }
    
    def get_orbital_info(self, calc_result: Dict[str, Any]) -> Tuple[float, float]:
        """
        Extract HOMO and LUMO energies from xTB calculation.
        
        Args:
            calc_result: Calculation results dictionary
            
        Returns:
            Tuple of (HOMO energy, LUMO energy) in atomic units
        """
        try:
            if 'orbital_energies' in calc_result:
                energies = calc_result['orbital_energies']
                # Assuming energies are sorted and we need to find HOMO/LUMO
                # Typically, occupied orbitals come first
                # For now, using simple heuristic: middle of energy range
                if len(energies) >= 2:
                    n_orbitals = len(energies)
                    homo_idx = n_orbitals // 2 - 1
                    lumo_idx = n_orbitals // 2
                    homo_energy = energies[homo_idx]
                    lumo_energy = energies[lumo_idx]
                    return homo_energy, lumo_energy
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not extract orbital energies: {e}")
    
    @property
    def name(self) -> str:
        """Return calculator description."""
        return f"xTB/{self.xtb_method}"
    
    def setup(self, atomic_numbers: np.ndarray, positions: np.ndarray,
              charge: int = 0, spin: int = 1):
        """
        Prepare molecule for xTB calculations.

        xTB/DFTB+ backend reads geometry from files written in single_point().
        This setup is a no-op and returns the provided data.
        """
        return atomic_numbers, positions, charge, spin