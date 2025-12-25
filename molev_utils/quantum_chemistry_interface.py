import sys
import os
import subprocess
import re
import importlib.util
from pathlib import Path

# Float parser to accept floats and scientific notation strings
def _safe_float(x, default=0.0):
    """Convert x to float. Accepts float, int, or numeric strings (including scientific notation).
       Returns default on failure."""
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        try:
            m = re.search(r'[-+]?\d*\.?\d+(?:[eE][+-]?\d+)?', str(x))
            if m:
                return float(m.group(0))
        except Exception:
            pass
    return default


class QuantumChemistryInterface:
    """Interface to quantum chemistry calculations via subprocess"""
    
    def __init__(self, calculator_type: str, calculator_kwargs: dict, 
                 method: str = "full_tensor", field_strength: float = 0.001,
                 properties: list = None, verbose: bool = False):
        """
        Initialize quantum chemistry interface.
        
        Args:
            calculator_type: Type of calculator (dft, semiempirical, etc.)
            calculator_kwargs: Calculator-specific arguments (functional, basis, etc.)
            method: Calculation method (full_tensor, finite_field, cphf)
            field_strength: Field strength for finite field methods
            properties: List of properties to calculate
            verbose: Enable verbose output
        """
        self.calculator_type = calculator_type.lower()
        self.calculator_kwargs = calculator_kwargs
        self.method = method.lower()
        self.field_strength = field_strength
        self.properties = properties or ['beta', 'dipole', 'homo_lumo_gap', 
                                         'transition_dipole', 'oscillator_strength', 
                                         'gamma', 'energy', 'alpha']
        self.verbose = verbose
        
        # Validate method
        valid_methods = ['full_tensor', 'finite_field', 'cphf']
        if self.method not in valid_methods:
            raise ValueError(f"Invalid method '{method}'. Must be one of: {valid_methods}")
        
        # Find quantum_chemistry directory
        self.qc_dir = self._find_qc_directory()
        if self.qc_dir is None:
            raise RuntimeError("Could not find quantum_chemistry directory")
        
        if self.verbose:
            print(f"Quantum chemistry directory: {self.qc_dir}")
            print(f"Calculation method: {self.method}")
    
    def _build_command(self, smiles: str, charge: int = 0, spin: int = 1) -> list:
        """Build the command line for quantum chemistry calculation"""
        cmd = [
            "python3",
            str(self.qc_dir / "main.py"),
            "--calculator", self.calculator_type,
            "--method", self.method,
            "--smiles", smiles,
            "--solvent", "none",
            "--field-strength", str(self.field_strength),
            "--charge", str(charge),
            "--spin", str(spin),
            "--properties"
        ] + self.properties
        
        # Add calculator-specific arguments
        if self.calculator_type == "dft":
            functional = self.calculator_kwargs.get('functional', 'B3LYP')
            basis = self.calculator_kwargs.get('basis', '6-31G')
            cmd.extend(["--functional", functional])
            cmd.extend(["--basis", basis])
        elif self.calculator_type == "semiempirical":
            se_method = self.calculator_kwargs.get('se_method', 'PM7')
            cmd.extend(["--se-method", se_method])
        elif self.calculator_type == "xtb":
            xtb_method = self.calculator_kwargs.get('xtb_method', 'GFN2-xTB')
            cmd.extend(["--xtb-method", xtb_method])
        
        # Add debug flag if needed
        if self.calculator_kwargs.get('debug_mopac', False):
            cmd.append("--debug-mopac")
        
        if self.verbose:
            cmd.append("--verbose")
        
        return cmd
    
    def _find_qc_directory(self):
        """Find the quantum_chemistry directory"""
        possible_paths = [
            Path.home() / "Molecular-Evolution" / "quantum_chemistry",
            Path(__file__).parent.parent.parent / "quantum_chemistry",
            Path.cwd() / ".." / ".." / "quantum_chemistry",
        ]
        
        for path in possible_paths:
            if path.exists() and (path / "main.py").exists():
                return path.resolve()
        
        return None
    
    def calculate(self, smiles: str, charge: int = 0, spin: int = 1) -> dict:
        """
        Run quantum chemistry calculation for a molecule.
        
        Args:
            smiles: SMILES string
            charge: Molecular charge
            spin: Spin multiplicity
            
        Returns:
            Dictionary with calculation results
        """
        cmd = self._build_command(smiles, charge, spin)
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=9999,
                cwd=str(self.qc_dir)
            )
            
            # Print full output in verbose mode
            if self.verbose:
                print(f"\n{'='*80}")
                print(f"Quantum Chemistry Calculation for: {smiles}")
                print(f"{'='*80}")
                print(result.stdout)
                if result.stderr:
                    print("STDERR:")
                    print(result.stderr)
                print(f"{'='*80}\n")
            
            return self._parse_output(result.stdout, result.stderr, smiles)
            
        except subprocess.TimeoutExpired:
            if self.verbose:
                print(f"Calculation timeout for {smiles}")
            return self._error_result(smiles, "Timeout")
        except Exception as e:
            if self.verbose:
                print(f"Calculation error for {smiles}: {e}")
            return self._error_result(smiles, str(e))
    
    def _parse_output(self, stdout: str, stderr: str, smiles: str) -> dict:
        """Parse the output from quantum chemistry calculation"""
        print(f"\nDEBUG QCI - _parse_output called for {smiles}")
        print(f"DEBUG QCI - File: {__file__}")
        result = {
            'smiles': smiles,
            'beta_vec': None,
            'beta_xxx': None,
            'beta_yyy': None,
            'beta_zzz': None,
            'beta_mean': None,
            'dipole_moment': None,
            'homo_lumo_gap': None,
            'total_energy': None,
            'alpha_mean': None,
            'gamma': None,
            'transition_dipole': None,
            'oscillator_strength': None,
            'error': None
        }
        
        # Check for actual errors (but ignore harmless warnings)
        combined_out = (stdout or "") + (stderr or "")

        # Ignore known harmless messages
        harmless_patterns = [
            "missing required dependencies",
            "overwritten attributes",
            "warning:"
        ]

        is_harmless = any(pattern in combined_out.lower() for pattern in harmless_patterns)

        if not is_harmless:
            # Check for real ERROR messages
            if "ERROR:" in combined_out or "error:" in combined_out:
                error_match = re.search(r'ERROR:\s*(.+)', combined_out, re.IGNORECASE)
                if error_match:
                    result['error'] = error_match.group(1).strip()
                    if self.verbose:
                        print(f"Calculation failed for {smiles}: {result['error']}")
                    return result

            # Check for convergence failure (only if no valid results section present)
            if ("did not converge" in stdout.lower() or "calculation failed" in stdout.lower()) and \
               "CALCULATION RESULTS" not in stdout:
                result['error'] = "Convergence failure"
                if self.verbose:
                    print(f"Convergence failure for {smiles}")
                return result
        
        # Parse beta values
        beta_mean = None
        beta_vec = None
        beta_components = {}

        try:
            # Generic patterns
            float_pat = r'([-+]?\d*\.?\d+(?:[eE][+-]?\d+)?)'
            # mean - Match ONLY "β mean:" with COLON (the actual result line, not debug output)
            # The actual QC output format is: "β mean:           6.750244e+03"
            mean_regex = re.compile(r'β\s+mean:\s+' + float_pat, re.IGNORECASE | re.UNICODE)
            m = mean_regex.search(stdout)
            if m:
                print(f"DEBUG REGEX: Matched string = '{m.group()}'")
                print(f"DEBUG REGEX: Extracted value = {m.group(1)}")
                beta_mean = float(m.group(1))

            # vector
            vec_regex = re.compile(r'(?:β|beta|b)[\s_]*(?:vector)[:\s]*' + float_pat, re.IGNORECASE | re.UNICODE)
            m = vec_regex.search(stdout)
            if m:
                beta_vec = float(m.group(1))

            # components xxx, yyy, zzz (allow either with or without β prefix)
            comp_regex = re.compile(r'(?:β|beta|b)?[\s_]*(xxx|yyy|zzz)[:\s]*' + float_pat, re.IGNORECASE | re.UNICODE)
            for comp_match in comp_regex.finditer(stdout):
                comp_name = comp_match.group(1).lower()
                comp_val = float(comp_match.group(2))
                if comp_name == 'xxx':
                    beta_components['beta_xxx'] = comp_val
                elif comp_name == 'yyy':
                    beta_components['beta_yyy'] = comp_val
                elif comp_name == 'zzz':
                    beta_components['beta_zzz'] = comp_val

            # Assign parsed values to result (leave None if not found)
            if beta_mean is not None:
                # Clamp negative beta_mean to 0.0
                if beta_mean < 0.0:
                    if self.verbose:
                        print(f"Clamping negative beta_mean ({beta_mean}) to 0.0 for {smiles}")
                    beta_mean = 0.0
                print(f"DEBUG: About to store beta_mean = {beta_mean}")
                result['beta_mean'] = beta_mean
                print(f"DEBUG: Stored in result dict, result['beta_mean'] = {result['beta_mean']}")
            if beta_vec is not None:
                result['beta_vec'] = beta_vec
            result.update({
                'beta_xxx': beta_components.get('beta_xxx', result['beta_xxx']),
                'beta_yyy': beta_components.get('beta_yyy', result['beta_yyy']),
                'beta_zzz': beta_components.get('beta_zzz', result['beta_zzz'])
            })
            print(f"DEBUG: After result.update(), result['beta_mean'] = {result.get('beta_mean')}")

        except Exception as e:
            if self.verbose:
                print(f"Beta parsing error for {smiles}: {e}")
            # Do not overwrite existing error state; leave beta fields as None

        # Parse dipole moment (in a.u.)
        dipole_match = re.search(r'Dipole moment:\s+([-+]?\d+\.\d+(?:e[+-]?\d+)?)\s+a\.u\.', stdout)
        if dipole_match:
            result['dipole_moment'] = float(dipole_match.group(1))
        
        # Parse HOMO-LUMO gap
        homo_lumo_match = re.search(r'HOMO-LUMO gap:\s+([-+]?\d+\.\d+(?:e[+-]?\d+)?)\s+eV', stdout)
        if homo_lumo_match:
            result['homo_lumo_gap'] = float(homo_lumo_match.group(1))
        
        # Parse total energy
        energy_match = re.search(r'Total energy:\s+([-+]?\d+\.\d+(?:e[+-]?\d+)?)\s+a\.u\.', stdout)
        if energy_match:
            result['total_energy'] = float(energy_match.group(1))
        
        # Parse alpha mean
        alpha_match = re.search(r'Alpha mean:\s+([-+]?\d+\.\d+e[+-]?\d+)\s+a\.u\.', stdout)
        if alpha_match:
            result['alpha_mean'] = float(alpha_match.group(1))
        
        # Parse gamma mean
        gamma_match = re.search(r'Gamma mean:\s+([-+]?\d+\.\d+e[+-]?\d+)\s+a\.u\.', stdout)
        if gamma_match:
            result['gamma'] = float(gamma_match.group(1))
        
        # Parse transition dipole
        trans_dip_match = re.search(r'Transition dipole:\s+([-+]?\d+\.\d+e[+-]?\d+)\s+a\.u\.', stdout)
        if trans_dip_match:
            result['transition_dipole'] = float(trans_dip_match.group(1))
        
        # Parse oscillator strength
        osc_match = re.search(r'Oscillator strength:\s+([-+]?\d+\.\d+e[+-]?\d+)', stdout)
        if osc_match:
            result['oscillator_strength'] = float(osc_match.group(1))

        # Calculate number of atoms from SMILES
        print(f"DEBUG QCI - About to calculate natoms for {smiles}")
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            if mol:
                result['natoms'] = mol.GetNumAtoms()
            else:
                result['natoms'] = 0
        except:
            result['natoms'] = 0
        print(f"DEBUG QCI - natoms = {result.get('natoms')}")

        # Calculate derived properties (ratios)
        print(f"DEBUG QCI - Calculating derived objectives for {smiles}")
        print(f"DEBUG: BEFORE derived calc, result['beta_mean'] = {result.get('beta_mean')}")
        print(f"  beta_mean: {result.get('beta_mean')}, gamma: {result.get('gamma')}")
        print(f"  total_energy: {result.get('total_energy')}, natoms: {result.get('natoms')}")
        print(f"  alpha_mean: {result.get('alpha_mean')}, homo_lumo_gap: {result.get('homo_lumo_gap')}")

        # beta_gamma_ratio = beta_mean / gamma
        if result.get('beta_mean') and result.get('gamma') and result['gamma'] != 0:
            result['beta_gamma_ratio'] = result['beta_mean'] / result['gamma']
            print(f"  Calculated beta_gamma_ratio: {result['beta_gamma_ratio']}")
        else:
            result['beta_gamma_ratio'] = 0.0
            print(f"  Set beta_gamma_ratio to 0.0 (condition not met)")

        # total_energy_atom_ratio = total_energy / natoms
        if result.get('total_energy') and result.get('natoms') and result['natoms'] > 0:
            result['total_energy_atom_ratio'] = result['total_energy'] / result['natoms']
            print(f"  Calculated total_energy_atom_ratio: {result['total_energy_atom_ratio']}")
        else:
            result['total_energy_atom_ratio'] = 0.0
            print(f"  Set total_energy_atom_ratio to 0.0 (condition not met)")

        # Range-based objectives: distance from target range (0 if within range)
        # alpha target range: [100, 500]
        alpha = result.get('alpha_mean', 0.0)
        if alpha < 100.0:
            result['alpha_range_distance'] = 100.0 - alpha
        elif alpha > 500.0:
            result['alpha_range_distance'] = alpha - 500.0
        else:
            result['alpha_range_distance'] = 0.0
        print(f"  Calculated alpha_range_distance: {result['alpha_range_distance']}")

        # homo_lumo_gap target range: [2.5, 3.5]
        homo_lumo_gap = result.get('homo_lumo_gap', 0.0)
        if homo_lumo_gap < 2.5:
            result['homo_lumo_gap_range_distance'] = 2.5 - homo_lumo_gap
        elif homo_lumo_gap > 3.5:
            result['homo_lumo_gap_range_distance'] = homo_lumo_gap - 3.5
        else:
            result['homo_lumo_gap_range_distance'] = 0.0
        print(f"  Calculated homo_lumo_gap_range_distance: {result['homo_lumo_gap_range_distance']}")

        print(f"DEBUG QCI - Final result keys: {list(result.keys())}")

        return result
    
    def _error_result(self, smiles: str, error: str) -> dict:
        """Return result dictionary for failed calculation"""
        return {
            'smiles': smiles,
            'beta_vec': 0.0,
            'beta_xxx': 0.0,
            'beta_yyy': 0.0,
            'beta_zzz': 0.0,
            'beta_mean': 0.0,
            'dipole_moment': 0.0,
            'homo_lumo_gap': 0.0,
            'total_energy': 0.0,
            'alpha_mean': 0.0,
            'gamma': 0.0,
            'transition_dipole': 0.0,
            'oscillator_strength': 0.0,
            'natoms': 0,
            'beta_gamma_ratio': 0.0,
            'total_energy_atom_ratio': 0.0,
            'alpha_range_distance': 100.0,  # Far from range
            'homo_lumo_gap_range_distance': 2.5,  # Far from range
            'error': error
        }
