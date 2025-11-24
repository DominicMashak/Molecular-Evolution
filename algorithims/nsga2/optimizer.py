"""
NSGA-II molecular optimizer with full multi-objective support
Supports: beta, natoms, homo_lumo_gap, energy, alpha, gamma, dipole, and more
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'molev_utils')))

import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import subprocess
import re
import importlib.util
from pathlib import Path
from molecule_generator import MoleculeGenerator
from dominance import dominates, fast_non_dominated_sort
from crowding import crowding_distance
from offspring import create_offspring
from database import update_molecule_database, save_molecule_database
from analysis import analyze_parent_child_performance
from plotting import plot_pareto_front
from results import save_results
from archive import BinnedParetoArchive
from pymoo.indicators.hv import Hypervolume

from performance import (
    PerformanceTracker, 
    PerformancePlotter, 
    HypervolumeCalculator
)

from typing import List

from stagnation import (
    StagnationAdaptiveMutation, 
    HybridStagnationStrategy,
    integrate_with_generator,
    update_generator_weights
)

logger = logging.getLogger(__name__)

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


# Helper function to dynamically load canonicalize_smiles
def _load_molecular_utils():
    """Dynamically load molecular.py from common quantum_chemistry locations."""
    candidates = [
        Path(__file__).parent.parent.parent / 'quantum_chemistry' / 'utils' / 'molecular.py',
        Path.home() / 'Molecular-Evolution' / 'quantum_chemistry' / 'utils' / 'molecular.py',
        Path.cwd() / '..' / '..' / 'quantum_chemistry' / 'utils' / 'molecular.py'
    ]
    for p in candidates:
        p = Path(p)
        if p.exists():
            spec = importlib.util.spec_from_file_location("molecular_utils", str(p))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    return None

_mol_utils = _load_molecular_utils()


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
        
        logger.info(f"Quantum chemistry directory: {self.qc_dir}")
        logger.info(f"Calculation method: {self.method}")
    
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
        
        if self.calculator_kwargs.get('debug_mopac', False):
            cmd.append("--debug-mopac")
        
        if self.verbose:
            cmd.append("--verbose")
        
        return cmd

    
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
                logger.info(f"\n{'='*80}")
                logger.info(f"Quantum Chemistry Calculation for: {smiles}")
                logger.info(f"{'='*80}")
                logger.info(result.stdout)
                if result.stderr:
                    logger.warning("STDERR:")
                    logger.warning(result.stderr)
                logger.info(f"{'='*80}\n")
            
            return self._parse_output(result.stdout, result.stderr, smiles)
            
        except subprocess.TimeoutExpired:
            logger.warning(f"Calculation timeout for {smiles}")
            return self._error_result(smiles, "Timeout")
        except Exception as e:
            logger.warning(f"Calculation error for {smiles}: {e}")
            return self._error_result(smiles, str(e))
    
    def _parse_output(self, stdout: str, stderr: str, smiles: str) -> dict:
        """Parse the output from quantum chemistry calculation"""
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
        
        combined_out = (stdout or "") + (stderr or "")
        if "ERROR:" in combined_out or "error:" in combined_out:
            if "missing required dependencies" in combined_out.lower():
                logger.warning(f"Ignoring dependency warning for {smiles}: 'Missing required dependencies'")
            else:
                error_match = re.search(r'ERROR:\s*(.+)', combined_out, re.IGNORECASE)
                if error_match:
                    result['error'] = error_match.group(1).strip()
                    logger.warning(f"Calculation failed for {smiles}: {result['error']}")
                    return result

        # Check for convergence failure
        if "did not converge" in stdout.lower() or "failed" in stdout.lower():
            result['error'] = "Convergence failure"
            logger.warning(f"Convergence failure for {smiles}")
            return result
        
        # Parse beta values
        # Accept both regular and scientific notation.
        beta_mean = None
        beta_vec = None
        beta_components = {}

        try:
            # Generic patterns
            float_pat = r'([-+]?\d*\.?\d+(?:[eE][+-]?\d+)?)'
            # mean
            mean_regex = re.compile(r'(?:β|beta|b)[\s_]*mean[:\s]*' + float_pat, re.IGNORECASE | re.UNICODE)
            m = mean_regex.search(stdout)
            if m:
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
                    logger.debug(f"Clamping negative beta_mean ({beta_mean}) to 0.0 for {smiles}")
                    beta_mean = 0.0
                result['beta_mean'] = beta_mean
            if beta_vec is not None:
                result['beta_vec'] = beta_vec
            result.update({
                'beta_xxx': beta_components.get('beta_xxx', result['beta_xxx']),
                'beta_yyy': beta_components.get('beta_yyy', result['beta_yyy']),
                'beta_zzz': beta_components.get('beta_zzz', result['beta_zzz'])
            })

        except Exception as e:
            logger.debug(f"Beta parsing error for {smiles}: {e}")
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
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            if mol:
                result['natoms'] = mol.GetNumAtoms()
            else:
                result['natoms'] = 0
        except:
            result['natoms'] = 0

        # Calculate derived properties (ratios)
        # beta_gamma_ratio = beta_mean / gamma
        if result.get('beta_mean') and result.get('gamma') and result['gamma'] != 0:
            result['beta_gamma_ratio'] = result['beta_mean'] / result['gamma']
        else:
            result['beta_gamma_ratio'] = 0.0

        # total_energy_atom_ratio = total_energy / natoms
        if result.get('total_energy') and result.get('natoms') and result['natoms'] > 0:
            result['total_energy_atom_ratio'] = result['total_energy'] / result['natoms']
        else:
            result['total_energy_atom_ratio'] = 0.0

        # Range-based objectives: distance from target range (0 if within range)
        # alpha target range: [100, 500]
        alpha = result.get('alpha_mean', 0.0)
        if alpha and alpha < 100.0:
            result['alpha_range_distance'] = 100.0 - alpha
        elif alpha and alpha > 500.0:
            result['alpha_range_distance'] = alpha - 500.0
        else:
            result['alpha_range_distance'] = 0.0

        # homo_lumo_gap target range: [2.5, 3.5]
        homo_lumo_gap = result.get('homo_lumo_gap', 0.0)
        if homo_lumo_gap and homo_lumo_gap < 2.5:
            result['homo_lumo_gap_range_distance'] = 2.5 - homo_lumo_gap
        elif homo_lumo_gap and homo_lumo_gap > 3.5:
            result['homo_lumo_gap_range_distance'] = homo_lumo_gap - 3.5
        else:
            result['homo_lumo_gap_range_distance'] = 0.0

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
            'error': error
        }


class NSGA2Optimizer:
    """NSGA-II with subprocess-based quantum chemistry calculations and full multi-objective support"""
    
    def __init__(self, 
                 generator: MoleculeGenerator,
                 calculator_type: str = "dft",
                 calculator_kwargs: dict = None,
                 method: str = "full_tensor",
                 pop_size: int = 100,
                 n_gen: int = 100,
                 plot_every: int = 10,
                 adaptive_mutation: bool = True,
                 output_dir: str = "nsga2_results",
                 objectives: list = None,
                 optimize_objectives: list = None,
                 reference_points: list = None,
                 seed: int = None,
                 initial_seeds: list = None,
                 archive_bins: int = 10,
                 archive_max_size: int = 1000,
                 n_parents: int = 50,
                 n_children: int = 50,
                 max_natoms: int = 50,
                 verbose: bool = False,
                 field_strength: float = 0.001,
                 enable_stagnation_response: bool = True,
                 stagnation_threshold: int = 5,
                 stagnation_strategy: str = "hybrid",
                 base_add_weight: float = 1.0,
                 weight_boost_factor: float = 2.0,
                 base_atoms_per_add: int = 1,
                 max_atoms_per_add: int = 5,
                 atoms_per_stagnation: float = 0.5,
                 base_addition_repeats: int = 1,
                 max_addition_repeats: int = 3):
        """Initialize optimizer with subprocess-based calculations"""
        
        self.generator = generator
        self.verbose = verbose
        self.field_strength = field_strength
        self.method = method
        
        # Initialize quantum chemistry interface
        self.qc_interface = QuantumChemistryInterface(
            calculator_type=calculator_type,
            calculator_kwargs=calculator_kwargs or {},
            method=method,
            field_strength=field_strength,
            verbose=verbose
        )
        
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.plot_every = plot_every
        self.adaptive_mutation = adaptive_mutation
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.objectives = objectives or ['beta', 'natoms']
        self.optimize_objectives = optimize_objectives or [('max', None), ('min', None)]
        self.max_natoms = max_natoms

        self.archive_bins = archive_bins
        self.archive_max_size = archive_max_size
        self.seed = seed
        self.initial_seeds = initial_seeds

        self.performance_tracker = PerformanceTracker(self.output_dir)
        self.performance_plotter = PerformancePlotter(self.output_dir)

        # Setup hypervolume reference point based on objectives
        if reference_points is not None:
            self.hv_reference = reference_points
        else:
            self.hv_reference = []
            for i, obj_name in enumerate(self.objectives):
                opt_type, target = self.optimize_objectives[i]
                if obj_name == 'natoms':
                    self.hv_reference.append(float(self.max_natoms))
                elif opt_type == 'max':
                    self.hv_reference.append(0.0)
                elif opt_type == 'min':
                    self.hv_reference.append(1000.0)  # Large value for minimization
                else:  # target
                    self.hv_reference.append(abs(target) * 10)
        
        # Remove direct Hypervolume usage; rely on archive for global HV and MOQD
        # self.hv_calculator = Hypervolume(ref_point=np.array(self.hv_reference))
        logger.info(f"Performance tracking initialized with reference point: {self.hv_reference}")

        self.archive = BinnedParetoArchive(
            n_bins=self.archive_bins, 
            max_size=self.archive_max_size, 
            optimize_objectives=self.optimize_objectives
        )
        
        if self.seed is not None:
            import random
            random.seed(self.seed)
            np.random.seed(self.seed)
        
        self.generation = 0
        self.population = []
        self.history = []
        self.all_molecules = []
        self.parent_child_stats = []
        self.n_parents = n_parents
        self.n_children = n_children
        
        # Initialize stagnation system
        self.enable_stagnation_response = enable_stagnation_response
        self.stagnation_strategy = stagnation_strategy
        
        if self.enable_stagnation_response:
            if stagnation_strategy == "hybrid":
                self.stagnation_system = HybridStagnationStrategy(
                    stagnation_threshold=stagnation_threshold,
                    base_add_weight=base_add_weight,
                    weight_boost_factor=weight_boost_factor,
                    base_atoms_per_add=base_atoms_per_add,
                    max_atoms_per_add=max_atoms_per_add,
                    base_addition_repeats=base_addition_repeats,
                    max_addition_repeats=max_addition_repeats
                )
            else:
                self.stagnation_system = StagnationAdaptiveMutation(
                    base_add_weight=base_add_weight,
                    base_atoms_per_add=base_atoms_per_add,
                    max_atoms_per_add=max_atoms_per_add,
                    stagnation_threshold=stagnation_threshold,
                    weight_boost_factor=weight_boost_factor,
                    atoms_per_stagnation=atoms_per_stagnation,
                    use_weight_boost=(stagnation_strategy in ["weight", "hybrid"]),
                    use_atom_boost=(stagnation_strategy in ["atoms", "hybrid"])
                )
            
            logger.info(f"Stagnation-adaptive mutation enabled (strategy: {stagnation_strategy})")
        else:
            self.stagnation_system = None
    
    def compute_objectives(self, smiles: str, atomic_numbers, positions) -> list:
        """
        Compute all objectives for a molecule using subprocess.

        Supports: beta, beta_mean, natoms, energy, dipole, homo_lumo_gap,
                 alpha, gamma, transition_dipole, oscillator_strength,
                 beta_gamma_ratio, total_energy_atom_ratio, alpha_range_distance,
                 homo_lumo_gap_range_distance
        """
        from rdkit import Chem
        
        # Get molecule info
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"Invalid SMILES: {smiles}")
            return [0.0 if opt[0] == 'max' else float('inf') 
                    for opt, _ in zip(self.optimize_objectives, self.objectives)]
        
        heavy_natoms = mol.GetNumAtoms()
        
        # Check atom count
        if heavy_natoms == 0 or heavy_natoms > self.max_natoms:
            logger.warning(f"Invalid number of atoms {heavy_natoms} for SMILES: {smiles}")
            return [0.0 if opt[0] == 'max' else float('inf') 
                    for opt, _ in zip(self.optimize_objectives, self.objectives)]
        
        # Run quantum chemistry calculation via subprocess
        qc_result = self.qc_interface.calculate(smiles)

        # DEBUG: Print all properties returned from QC calculation
        logger.info(f"DEBUG - QC Result for {smiles}:")
        for key, value in qc_result.items():
            logger.info(f"  {key}: {value}")

        logger.info(f"DEBUG - Objectives we're evolving: {self.objectives}")

        # Build objectives list based on requested objectives
        obj_values = []
        for obj_name in self.objectives:
            if obj_name in ['beta', 'beta_mean']:
                # Use robust conversion for scientific notation and strings
                value = _safe_float(qc_result.get('beta_mean', 0.0), default=0.0)
                # Clamp extreme/parsing failures
                if abs(value) > 1e10:
                    value = 0.0
                # Clamp negative beta_mean to 0.0
                if value < 0.0:
                    logger.debug(f"Clamping negative objective beta_mean ({value}) to 0.0 for {smiles}")
                    value = 0.0
                obj_values.append(float(value))
            
            elif obj_name == 'natoms':
                obj_values.append(heavy_natoms)
            
            elif obj_name == 'energy':
                value = _safe_float(qc_result.get('total_energy', 0.0), default=0.0)
                obj_values.append(float(value))
            
            elif obj_name == 'dipole':
                value = _safe_float(qc_result.get('dipole_moment', 0.0), default=0.0)
                obj_values.append(float(value))
            
            elif obj_name == 'homo_lumo_gap':
                value = _safe_float(qc_result.get('homo_lumo_gap', 0.0), default=0.0)
                obj_values.append(float(value))
            
            elif obj_name == 'alpha':
                value = _safe_float(qc_result.get('alpha_mean', 0.0), default=0.0)
                obj_values.append(float(value))
            
            elif obj_name == 'gamma':
                value = _safe_float(qc_result.get('gamma', 0.0), default=0.0)
                obj_values.append(float(value))
            
            elif obj_name == 'transition_dipole':
                value = _safe_float(qc_result.get('transition_dipole', 0.0), default=0.0)
                obj_values.append(float(value))
            
            elif obj_name == 'oscillator_strength':
                value = _safe_float(qc_result.get('oscillator_strength', 0.0), default=0.0)
                obj_values.append(float(value))

            # Derived objectives
            elif obj_name == 'beta_gamma_ratio':
                value = _safe_float(qc_result.get('beta_gamma_ratio', 0.0), default=0.0)
                obj_values.append(float(value))

            elif obj_name == 'total_energy_atom_ratio':
                value = _safe_float(qc_result.get('total_energy_atom_ratio', 0.0), default=0.0)
                obj_values.append(float(value))

            elif obj_name == 'alpha_range_distance':
                value = _safe_float(qc_result.get('alpha_range_distance', 0.0), default=0.0)
                obj_values.append(float(value))

            elif obj_name == 'homo_lumo_gap_range_distance':
                value = _safe_float(qc_result.get('homo_lumo_gap_range_distance', 0.0), default=0.0)
                obj_values.append(float(value))

            else:
                logger.warning(f"Unknown objective: {obj_name}, using 0.0")
                obj_values.append(0.0)
        
        # After computing objectives, add to MAP-Elites archive
        obj_dict = {
            'beta': obj_values[0] if len(obj_values) > 0 else 0.0,  # Assuming beta is first objective
            'num_atoms_bin': min(9, heavy_natoms // 5),
            'homo_lumo_gap_bin': min(9, int(qc_result.get('homo_lumo_gap', 0.0) or 0.0)),
            'num_atoms': heavy_natoms,
            'homo_lumo_gap': qc_result.get('homo_lumo_gap', 0.0) or 0.0
        }
        
        return obj_values
    
    def create_individual(self, smiles: str, generation: int) -> 'Individual':
        """Create an Individual from SMILES string"""
        from individual import Individual
        from rdkit import Chem
        from rdkit.Chem import AllChem

        # Try to canonicalize using dynamically-loaded molecular utils if available
        global _mol_utils
        if _mol_utils is None:
            _mol_utils = _load_molecular_utils()
        if _mol_utils and hasattr(_mol_utils, 'canonicalize_smiles'):
            try:
                smiles = _mol_utils.canonicalize_smiles(smiles)
            except Exception:
                pass
        else:
            try:
                mol_tmp = Chem.MolFromSmiles(smiles)
                if mol_tmp:
                    smiles = Chem.MolToSmiles(mol_tmp, canonical=True)
            except Exception:
                pass
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"Invalid SMILES: {smiles}")
            return Individual(
                smiles=smiles,
                natoms=0,
                beta_surrogate=0.0,
                generation=generation
            )

        heavy_natoms = mol.GetNumAtoms()
        mol = Chem.AddHs(mol)
        atomic_numbers = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()])
        
        # Embed molecule for geometry
        positions = []
        try:
            embed_seed = self.seed if self.seed is not None else 42
            embed_result = AllChem.EmbedMolecule(mol, randomSeed=embed_seed)
            if embed_result == 0:
                AllChem.MMFFOptimizeMolecule(mol)
                conf = mol.GetConformer()
                for atom in mol.GetAtoms():
                    pos = conf.GetAtomPosition(atom.GetIdx())
                    positions.append([pos.x, pos.y, pos.z])
            else:
                logger.warning(f"Failed to embed molecule {smiles}")
                for i in range(mol.GetNumAtoms()):
                    positions.append([0.0, 0.0, float(i) * 1.5])
        except Exception as e:
            logger.warning(f"Error processing molecule {smiles}: {e}")
            for i in range(mol.GetNumAtoms()):
                positions.append([0.0, 0.0, float(i) * 1.5])
        
        positions = np.array(positions)
        
        # Compute objectives using subprocess
        objectives = self.compute_objectives(smiles, atomic_numbers, positions)
        
        # Get additional properties from QC result
        qc_result = self.qc_interface.calculate(smiles)
        
        # Find beta value in objectives (if present)
        beta_val = 0.0
        if 'beta' in self.objectives or 'beta_mean' in self.objectives:
            beta_idx = self.objectives.index('beta') if 'beta' in self.objectives else self.objectives.index('beta_mean')
            beta_val = float(objectives[beta_idx]) if len(objectives) > beta_idx else 0.0
        
        # Find natoms value in objectives (if present)
        natoms_val = heavy_natoms
        if 'natoms' in self.objectives:
            natoms_idx = self.objectives.index('natoms')
            natoms_val = int(objectives[natoms_idx]) if len(objectives) > natoms_idx else heavy_natoms
        
        return Individual(
            smiles=smiles,
            objectives=objectives,
            beta_surrogate=beta_val,
            natoms=natoms_val,
            generation=generation,
            homo_lumo_gap=qc_result.get('homo_lumo_gap', 0.0) or 0.0,
            transition_dipole=qc_result.get('transition_dipole', 0.0) or 0.0,
            oscillator_strength=qc_result.get('oscillator_strength', 0.0) or 0.0,
            gamma=qc_result.get('gamma', 0.0) or 0.0,
            alpha_mean=qc_result.get('alpha_mean', 0.0) or 0.0
        )
    
    def create_offspring_with_adaptive_mutation(self, parents):
        """Create offspring with stagnation-adaptive mutations"""
        children = []
        
        if self.stagnation_system:
            if self.stagnation_strategy == "hybrid":
                n_repeats = self.stagnation_system.current_addition_repeats
            else:
                n_repeats = 1
        else:
            n_repeats = 1
        
        while len(children) < self.n_children:
            if not parents:
                break
            tournament_size = min(3, len(parents))
            tournament = np.random.choice(parents, tournament_size, replace=False).tolist()
            winner = min(tournament, key=lambda x: (x.rank, -x.crowding_distance))
            
            mutated_smiles = winner.smiles
            success = False
            
            for repeat in range(n_repeats):
                try:
                    temp_smiles = self.generator.mutate_multiple(mutated_smiles)
                    
                    if temp_smiles and self.generator.validate_molecule(temp_smiles):
                        mutated_smiles = temp_smiles
                        success = True
                        logger.debug(f"  Mutation repeat {repeat+1}/{n_repeats} successful")
                    else:
                        break
                except Exception as e:
                    logger.debug(f"  Mutation repeat {repeat+1}/{n_repeats} failed: {e}")
                    break
            
            if success and mutated_smiles != winner.smiles:
                child = self.create_individual(mutated_smiles, self.generation + 1)
                children.append(child)
            elif len(children) < self.n_children:
                try:
                    simple_mutated = self.generator.mutate_multiple(winner.smiles)
                    if simple_mutated and self.generator.validate_molecule(simple_mutated):
                        child = self.create_individual(simple_mutated, self.generation + 1)
                        children.append(child)
                except Exception:
                    pass
        
        return children
    
    # Delegate methods
    dominates = staticmethod(dominates)
    fast_non_dominated_sort = staticmethod(fast_non_dominated_sort)
    crowding_distance = staticmethod(crowding_distance)
    analyze_parent_child_performance = analyze_parent_child_performance
    update_molecule_database = update_molecule_database
    save_molecule_database = save_molecule_database
    plot_pareto_front = plot_pareto_front
    save_results = save_results

    def run(self):
        """Main optimization loop"""
        logger.info(f"Starting NSGA-II optimization")
        logger.info(f"Pop size: {self.pop_size}, Generations: {self.n_gen}")
        logger.info(f"Calculator: {self.qc_interface.calculator_type}")
        logger.info(f"Method: {self.method}") 
        logger.info(f"Objectives: {self.objectives}")
        logger.info(f"Optimization: {[opt[0] for opt in self.optimize_objectives]}")
        if self.enable_stagnation_response:
            logger.info(f"Stagnation response: ENABLED ({self.stagnation_strategy} strategy)")
        
        # Initialize population
        logger.info("Generating initial population...")
        if self.initial_seeds:
            initial_smiles = self.initial_seeds[:]
            if len(initial_smiles) < self.pop_size:
                additional = self.generator.generate_initial_population(
                    self.pop_size - len(initial_smiles),
                    save_to_file=True,
                    seed_number=self.seed,
                    algorithm_name="nsga2"
                )
                initial_smiles.extend(additional)
        else:
            initial_smiles = self.generator.generate_initial_population(
                self.pop_size,
                save_to_file=True,
                seed_number=self.seed,
                algorithm_name="nsga2"
            )
        
        logger.info("Calculating properties for initial population...")
        self.population = []
        for i, smiles in enumerate(initial_smiles):
            if i % 10 == 0:
                logger.info(f"  Processing molecule {i+1}/{self.pop_size}")
            ind = self.create_individual(smiles, 0)
            self.population.append(ind)
            
            # Log all objectives
            obj_str = ", ".join([f"{name}={val:.6e}" if isinstance(val, float) and abs(val) < 0.01 
                                else f"{name}={val}" 
                                for name, val in zip(self.objectives, ind.objectives)])
            logger.info(f"  {smiles}: {obj_str}")
        
        self.update_molecule_database(self.population)
        
        for ind in self.population:
            self.archive.add(ind)

        # Track initial population
        initial_fronts = self.fast_non_dominated_sort(self.population, self.optimize_objectives)
        # Use global hypervolume and MOQD from archive
        initial_global_hv = self.archive.compute_global_hypervolume()
        initial_moqd = self.archive.compute_moqd_score()
        self.performance_tracker.update(0, self.population, initial_fronts, initial_global_hv, initial_moqd)
        
        # Main loop
        for gen in range(self.n_gen):
            self.generation = gen
            logger.info(f"\n{'='*60}")
            logger.info(f"Generation {gen}/{self.n_gen}")
            
            # Get best value for first objective (typically beta)
            if self.population:
                best_obj0 = max(ind.objectives[0] for ind in self.population) if self.optimize_objectives[0][0] == 'max' else min(ind.objectives[0] for ind in self.population)
            else:
                best_obj0 = 0.0
            
            # Update stagnation system (using first objective)
            if self.enable_stagnation_response and self.stagnation_system:
                mutation_stats = self.stagnation_system.update(gen, best_obj0)
                
                if mutation_stats['is_stagnant']:
                    if hasattr(self.stagnation_system, 'get_mutation_weights'):
                        adjusted_weights = self.stagnation_system.get_mutation_weights(
                            self.generator.mutation_weights
                        )
                        self.generator.mutation_weights = adjusted_weights
                    
                    logger.warning(f"STAGNATION: {mutation_stats['stagnation_duration']} generations")
                    logger.info(f"   Current best {self.objectives[0]}: {best_obj0:.6e}")
            
            # Selection
            fronts = self.fast_non_dominated_sort(self.population, self.optimize_objectives)
            parents = []
            for front in fronts:
                if len(parents) + len(front) <= self.n_parents:
                    parents.extend(front)
                else:
                    self.crowding_distance(front, self.optimize_objectives)
                    front.sort(key=lambda x: -x.crowding_distance)
                    parents.extend(front[:self.n_parents - len(parents)])
                    break
            
            # Create offspring
            if self.enable_stagnation_response:
                children = self.create_offspring_with_adaptive_mutation(parents)
            else:
                children = []
                while len(children) < self.n_children:
                    if not parents:
                        break
                    tournament_size = min(3, len(parents))
                    tournament = np.random.choice(parents, tournament_size, replace=False).tolist()
                    winner = min(tournament, key=lambda x: (x.rank, -x.crowding_distance))
                    mutated_smiles = self.generator.mutate_multiple(winner.smiles)
                    if mutated_smiles and self.generator.validate_molecule(mutated_smiles):
                        child = self.create_individual(mutated_smiles, self.generation + 1)
                        children.append(child)
            
            logger.info(f"Generated {len(children)} offspring")
            logger.info("Calculating properties for offspring...")
            for i, child in enumerate(children):
                if i % 10 == 0:
                    logger.info(f"  Processing child {i+1}/{len(children)}")
                
                # Log all objectives
                obj_str = ", ".join([f"{name}={val:.6e}" if isinstance(val, float) and abs(val) < 0.01 
                                    else f"{name}={val}" 
                                    for name, val in zip(self.objectives, child.objectives)])
                logger.info(f"  {child.smiles}: {obj_str}")

            # Update database with new children only (parents already in database from previous generations)
            self.update_molecule_database(children)
            combined_population = parents + children
            self.analyze_parent_child_performance(parents, children)
            
            # Environmental selection
            fronts = self.fast_non_dominated_sort(combined_population, self.optimize_objectives)
            
            new_population = []
            for front in fronts:
                if len(new_population) + len(front) <= self.pop_size:
                    new_population.extend(front)
                else:
                    self.crowding_distance(front, self.optimize_objectives)
                    front.sort(key=lambda x: -x.crowding_distance)
                    new_population.extend(front[:self.pop_size - len(new_population)])
                    break
            
            self.population = new_population

            # Calculate metrics using global hypervolume and MOQD
            fronts = self.fast_non_dominated_sort(self.population, self.optimize_objectives)
            global_hv = self.archive.compute_global_hypervolume()
            moqd = self.archive.compute_moqd_score()
            self.performance_tracker.update(gen + 1, self.population, fronts, global_hv, moqd)

            logger.info(f"Global Hypervolume: {global_hv:.6f}, MOQD: {moqd:.6f}")
            
            # Log pareto front details
            best_front = fronts[0] if fronts else []
            logger.info(f"Front 0 size: {len(best_front)}")
            
            if best_front:
                # Log statistics for each objective
                for i, obj_name in enumerate(self.objectives):
                    obj_vals = [ind.objectives[i] for ind in best_front]
                    if self.optimize_objectives[i][0] == 'max':
                        best_val = max(obj_vals)
                        logger.info(f"Best {obj_name}: {best_val:.6e}" if abs(best_val) < 0.01 else f"Best {obj_name}: {best_val}")
                    else:
                        best_val = min(obj_vals)
                        logger.info(f"Best {obj_name}: {best_val:.6e}" if abs(best_val) < 0.01 else f"Best {obj_name}: {best_val}")
                
                # Log top 3 molecules by first objective
                sorted_front = sorted(best_front, key=lambda x: -x.objectives[0] if self.optimize_objectives[0][0] == 'max' else x.objectives[0])[:3]
                logger.info(f"Top 3 molecules by {self.objectives[0]}:")
                for i, ind in enumerate(sorted_front, 1):
                    obj_str = ", ".join([f"{name}={val:.6e}" if isinstance(val, float) and abs(val) < 0.01 
                                        else f"{name}={val}" 
                                        for name, val in zip(self.objectives, ind.objectives)])
                    logger.info(f"  {i}. {ind.smiles}")
                    logger.info(f"     {obj_str}")
            
            # Periodic actions
            if gen % self.plot_every == 0:
                self.plot_pareto_front(self.population, gen)
            
            if gen % 5 == 0:
                self.save_molecule_database()
            
            if gen % 10 == 0:
                # Log mutation statistics
                success_rates = self.generator.get_mutation_success_rates()
                logger.info("Mutation success rates:")
                for name, rate in success_rates.items():
                    logger.info(f"  {name}: {rate:.2%}")
                
                # Log stagnation statistics
                if self.enable_stagnation_response and self.stagnation_system:
                    stag_stats = self.stagnation_system.get_statistics() if hasattr(
                        self.stagnation_system, 'get_statistics'
                    ) else {}
                    if stag_stats:
                        logger.info("Stagnation statistics:")
                        logger.info(f"  Best fitness ever: {stag_stats.get('best_fitness_ever', 0):.6e}")
                        logger.info(f"  Stagnation counter: {stag_stats.get('stagnation_counter', 0)}")
                        logger.info(f"  Total interventions: {stag_stats.get('total_interventions', 0)}")
        
        # Final actions
        self.plot_pareto_front(self.population, self.n_gen)
        self.save_results()
        
        logger.info("\n" + "="*60)
        logger.info("OPTIMIZATION COMPLETE")
        logger.info("="*60)
        
        # Final stagnation report
        if self.enable_stagnation_response and self.stagnation_system:
            logger.info("\nStagnation Response Summary:")
            stats = self.stagnation_system.get_statistics() if hasattr(
                self.stagnation_system, 'get_statistics'
            ) else {}
            if stats:
                logger.info(f"  Total interventions: {stats.get('total_interventions', 0)}")
                logger.info(f"  Best fitness achieved: {stats.get('best_fitness_ever', 0):.6e}")
                
                if 'intervention_history' in stats and stats['intervention_history']:
                    logger.info("  Recent interventions:")
                    for interv in stats['intervention_history'][-5:]:
                        logger.info(f"    Gen {interv['generation']}: "
                                  f"duration={interv['stagnation_duration']}, "
                                  f"beta={interv['best_fitness']:.6e}")
        
        fronts = self.fast_non_dominated_sort(self.population, self.optimize_objectives)
        if fronts and fronts[0]:
            sorted_front = sorted(fronts[0], key=lambda x: -x.objectives[0] if self.optimize_objectives[0][0] == 'max' else x.objectives[0])[:10]
            logger.info(f"\nTop 10 molecules by {self.objectives[0]}:")
            for i, ind in enumerate(sorted_front, 1):
                obj_str = ", ".join([f"{name}={val:.6e}" if isinstance(val, float) and abs(val) < 0.01 
                                    else f"{name}={val}" 
                                    for name, val in zip(self.objectives, ind.objectives)])
                logger.info(f"{i:2d}. {ind.smiles}")
                logger.info(f"    {obj_str}")
        
        # Periodic plotting
        if self.n_gen % self.plot_every == 0:
            self.performance_plotter.plot_convergence(
                self.performance_tracker, 
                f'convergence_gen_{self.n_gen:03d}.pdf'
            )
        
        # Create parallel coordinates plot
        if self.n_gen in [0, self.n_gen // 4, self.n_gen // 2, 3 * self.n_gen // 4]:
            self.performance_plotter.plot_parallel_coordinates(
                self.population, self.n_gen
            )
    
    def save_results(self):
        """Save results including all objectives"""
        from individual import Individual

        # Save pareto front
        all_inds = [Individual(smiles=mol['smiles'], objectives=mol.get('objectives', []), 
                              generation=mol['generation']) 
                   for mol in self.all_molecules if 'objectives' in mol]
        
        if not all_inds:
            # Fallback for old format
            all_inds = [Individual(smiles=mol['smiles'], 
                                  natoms=mol.get('natoms', 0), 
                                  beta_surrogate=mol.get('beta_surrogate', 0.0), 
                                  generation=mol['generation']) 
                       for mol in self.all_molecules]
        
        fronts = self.fast_non_dominated_sort(all_inds, self.optimize_objectives)
        results = []
        for ind in fronts[0] if fronts else []:
            result_dict = {
                'smiles': ind.smiles,
                'objectives': ind.objectives,
                'generation': ind.generation
            }
            # Add named objective values for clarity
            for i, obj_name in enumerate(self.objectives):
                if i < len(ind.objectives):
                    result_dict[obj_name] = ind.objectives[i]
            results.append(result_dict)
        
        pareto_file = self.output_dir / "pareto_front_molecules.json"
        with open(pareto_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save top 10 by first objective
        sorted_molecules = sorted(
            [m for m in self.all_molecules if 'objectives' in m and m['objectives']], 
            key=lambda x: -x['objectives'][0] if self.optimize_objectives[0][0] == 'max' else x['objectives'][0]
        )[:10]
        
        top_10_file = self.output_dir / f"top_10_{self.objectives[0]}_molecules.json"
        with open(top_10_file, 'w') as f:
            json.dump(sorted_molecules, f, indent=2)
        
        # Save other statistics
        stats_file = self.output_dir / "parent_child_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(self.parent_child_stats, f, indent=2)
        
        weights_file = self.output_dir / "final_mutation_weights.json"
        with open(weights_file, 'w') as f:
            json.dump(self.generator.mutation_weights, f, indent=2)
        
        archive_file = self.output_dir / "archive.json"
        with open(archive_file, 'w') as f:
            archive_data = [
                {
                    'smiles': ind.smiles,
                    'objectives': ind.objectives,
                    'homo_lumo_gap': ind.homo_lumo_gap,
                    'transition_dipole': ind.transition_dipole,
                    'oscillator_strength': ind.oscillator_strength,
                    'gamma': ind.gamma,
                    'alpha_mean': ind.alpha_mean,
                    'generation': ind.generation
                }
                for ind in self.archive.get_all_individuals()
            ]
            json.dump(archive_data, f, indent=2)
        
        # Save stagnation statistics
        if self.enable_stagnation_response and self.stagnation_system:
            stag_stats = self.stagnation_system.get_statistics() if hasattr(
                self.stagnation_system, 'get_statistics'
            ) else {}
            if stag_stats:
                stagnation_file = self.output_dir / "stagnation_statistics.json"
                with open(stagnation_file, 'w') as f:
                    json.dump(stag_stats, f, indent=2)
                logger.info(f"Saved stagnation statistics to {stagnation_file}")

        # Save and plot final performance metrics
        self.performance_tracker.save()
        self.performance_plotter.plot_convergence(
            self.performance_tracker, 
            'final_convergence.pdf'
        )

        # Create final parallel coordinates plot
        self.performance_plotter.plot_parallel_coordinates(
            self.population, 
            self.n_gen, 
            'final_parallel_coordinates.pdf'
        )
        
        # Log final performance summary
        logger.info("\n" + "="*60)
        logger.info("PERFORMANCE SUMMARY")
        logger.info("="*60)
        
        metrics = self.performance_tracker.metrics
        if metrics['hypervolume']:
            initial_hv = metrics['hypervolume'][0]
            final_hv = metrics['hypervolume'][-1]
            hv_improvement = (final_hv - initial_hv) / (initial_hv + 1e-10) * 100
            logger.info(f"Global Hypervolume: {initial_hv:.6f} → {final_hv:.6f} ({hv_improvement:+.1f}%)")
        
        # Save all_molecules_database.json
        molecules_file = self.output_dir / "all_molecules_database.json"
        with open(molecules_file, 'w') as f:
            json.dump(self.all_molecules, f, indent=2)

    @staticmethod
    def compare_runs(run_dirs: List[str], output_dir: str = 'comparison_plots'):
        """Compare performance across multiple optimization runs"""
        from pathlib import Path
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        trackers = {}
        for run_dir in run_dirs:
            run_path = Path(run_dir)
            metrics_file = run_path / 'performance_metrics.json'
            if metrics_file.exists():
                tracker = PerformanceTracker(run_path)
                with open(metrics_file, 'r') as f:
                    tracker.metrics = json.load(f)
                trackers[run_path.name] = tracker
        
        if trackers:
            plotter = PerformancePlotter(output_path)
            plotter.plot_hypervolume_comparison(trackers)
            
            # Create comparison statistics
            comparison_stats = {}
            for name, tracker in trackers.items():
                metrics = tracker.metrics
                comparison_stats[name] = {
                    'final_hypervolume': metrics['hypervolume'][-1] if metrics['hypervolume'] else 0,
                    'final_max_beta': metrics['max_beta'][-1] if metrics['max_beta'] else 0,
                    'final_pareto_size': metrics['pareto_size'][-1] if metrics['pareto_size'] else 0,
                    'generations': len(metrics['generation'])
                }
            
            # Save comparison statistics
            stats_file = output_path / 'comparison_statistics.json'
            with open(stats_file, 'w') as f:
                json.dump(comparison_stats, f, indent=2)
            
            logger.info(f"Saved comparison plots and statistics to {output_path}")
        else:
            logger.warning("No valid performance metrics found in specified directories")