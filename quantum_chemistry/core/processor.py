#!/usr/bin/env python3
"""
Main processor for orchestrating molecular property calculations.
"""

import time
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
from datetime import datetime
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
quantum_chem_dir = os.path.dirname(current_dir)
sys.path.insert(0, quantum_chem_dir)

from core.base import Calculator, HyperpolarizabilityMethod, CalculationResult
from core.types import (
    Molecule, MoleculeResult, CalculatorConfig, MethodConfig,
    CalculatorType, MethodType
)
from utils.molecular import smiles_to_geometry, canonicalize_smiles, get_molecular_descriptors


class Processor:
    """
    Main processor for molecular property calculations.
    
    This class orchestrates the calculation pipeline, handling molecule
    processing, calculator setup, and result aggregation.
    """
    
    def __init__(self, calculator: Calculator, method: HyperpolarizabilityMethod,
                 verbose: bool = False):
        """
        Initialize processor with calculator and method.
        
        Args:
            calculator: Calculator backend
            method: Hyperpolarizability calculation method
            verbose: Print detailed output
        """
        self.calculator = calculator
        self.method = method
        self.verbose = verbose
    
    @classmethod
    def from_config(cls, calc_config: CalculatorConfig, method_config: Optional[MethodConfig],
                    verbose: bool = False) -> 'Processor':
        """
        Create processor from configuration objects.
        
        Args:
            calc_config: Calculator configuration
            method_config: Method configuration
            verbose: Print detailed output
            
        Returns:
            Configured Processor instance
        """
        # Import calculator implementations
        calculator = None  # Initialize to None
        
        if calc_config.calculator_type == CalculatorType.DFT:
            from calculators.dft import DFTCalculator
            calculator = DFTCalculator(**calc_config.to_dict())

        elif calc_config.calculator_type == CalculatorType.XTB:
            from calculators.xtb import XTBCalculator
            calculator = XTBCalculator(**calc_config.to_dict())
            
        elif calc_config.calculator_type == CalculatorType.CC:
            from calculators.cc import CCCalculator
            calculator = CCCalculator(**calc_config.to_dict())
            
        elif calc_config.calculator_type == CalculatorType.SEMIEMPIRICAL:
            from calculators.semiempirical import SemiEmpiricalCalculator
            calculator = SemiEmpiricalCalculator(**calc_config.to_dict())
            
        else:
            raise ValueError(f"Unknown calculator type: {calc_config.calculator_type}")
        
        # Verify calculator was created
        if calculator is None:
            raise RuntimeError(f"Failed to create calculator for type: {calc_config.calculator_type}")
        
        # For PM6/PM7, use a dummy method since MOPAC handles hyperpolarizability
        if (calc_config.calculator_type == CalculatorType.SEMIEMPIRICAL and 
            calc_config.se_method in ("PM6", "PM7")):
            class DummyMethod:
                def __init__(self, calculator):
                    self.calculator = calculator
                    self.name = "mopac"
                
                def calculate(self, atomic_numbers, coords, charge, spin, **kwargs):
                    # Directly call calculator's single_point method
                    result = self.calculator.single_point(atomic_numbers, coords, charge, spin)
                    # Wrap result in a compatible object
                    class Result:
                        def __init__(self, calc_result):
                            self.beta_vec = calc_result.get('beta_vec')
                            self.beta_xxx = calc_result.get('beta_xxx')
                            self.beta_yyy = calc_result.get('beta_yyy')
                            self.beta_zzz = calc_result.get('beta_zzz')
                            self.beta_mean = calc_result.get('beta_mean')
                            self.dipole_moment = calc_result.get('dipole')
                            self.homo_lumo_gap = calc_result.get('homo_lumo_gap')
                            self.total_energy = calc_result.get('total_energy')
                            self.beta_tensor = calc_result.get('beta_tensor')
                            self.alpha_tensor = calc_result.get('alpha_tensor')
                            self.alpha_mean = calc_result.get('alpha_mean')
                            self.error = calc_result.get('error')
                    
                    return Result(result)
            
            method = DummyMethod(calculator)
        else:
            # Import method implementations — ensure method_config is valid
            if method_config is None:
                raise ValueError("method_config is None for selected calculator; a method must be provided.")
            
            if method_config.method_type == MethodType.FINITE_FIELD:
                from methods.finite_field import FiniteFieldMethod
                method = FiniteFieldMethod(calculator, **method_config.to_dict())
            elif method_config.method_type == MethodType.FULL_TENSOR:
                from methods.full_tensor import FullTensorMethod
                method = FullTensorMethod(calculator, **method_config.to_dict())
            elif method_config.method_type == MethodType.CPHF:
                from methods.cphf import CPHFMethod
                method = CPHFMethod(calculator, **method_config.to_dict())
            else:
                raise ValueError(f"Unknown method type: {method_config.method_type}")
        
        return cls(calculator, method, verbose)
    
    def process_molecule(self, smiles: str, charge: int = 0, 
                        spin: int = 1) -> MoleculeResult:
        """
        Process a single molecule to calculate hyperpolarizability.
        
        Args:
            smiles: SMILES string
            charge: Molecular charge
            spin: Spin multiplicity (2S+1)
            
        Returns:
            MoleculeResult with calculated properties
        """
        start_time = time.time()
        
        # Initialize result
        result = MoleculeResult(
            smiles=smiles,
            method=f"{self.calculator.name}/{self.method.name}"
        )
        
        try:
            # Canonicalize SMILES
            canonical_smiles = canonicalize_smiles(smiles)
            
            # Generate 3D geometry
            coords, formula, rdkit_mol, atomic_numbers = smiles_to_geometry(canonical_smiles)
            
            if coords is None:
                result.error = "Failed to generate 3D geometry"
                result.wall_time = time.time() - start_time
                return result
            
            result.formula = formula
            
            # Calculate hyperpolarizability
            # Call method calculate with the standard arguments.
            calc_result = self.method.calculate(
                atomic_numbers, coords, charge, spin
            )
            
            # Transfer results - include all possible properties
            for key in ['beta_vec', 'beta_xxx', 'beta_yyy', 'beta_zzz', 'beta_mean',
                       'dipole_moment', 'homo_lumo_gap', 'total_energy', 'error',
                       'transition_dipole', 'oscillator_strength', 'gamma', 'alpha_mean']:
                if hasattr(calc_result, key):
                    setattr(result, key, getattr(calc_result, key))
            
            # Handle tensor data
            if hasattr(calc_result, 'beta_tensor') and calc_result.beta_tensor is not None:
                result.beta_tensor = calc_result.beta_tensor.tolist()
            if hasattr(calc_result, 'alpha_tensor') and calc_result.alpha_tensor is not None:
                result.alpha_tensor = calc_result.alpha_tensor.tolist()
            
        except Exception as e:
            result.error = str(e)
            if self.verbose:
                import traceback
                print(f"Error processing {smiles}: {e}")
                print(traceback.format_exc())
        
        result.wall_time = time.time() - start_time
        
        return result
    
    def process_batch(self, input_file: str, output_file: str,
                     max_molecules: Optional[int] = None,
                     save_interval: int = 10,
                     charge_col: Optional[str] = None,
                     spin_col: Optional[str] = None) -> pd.DataFrame:
        """
        Process molecules from CSV file in batch mode.
        
        Args:
            input_file: Input CSV file with SMILES column
            output_file: Output CSV file for results
            max_molecules: Maximum number of molecules to process
            save_interval: Save results every N molecules
            charge_col: Column name for molecular charges
            spin_col: Column name for spin multiplicities
            
        Returns:
            DataFrame with calculation results
        """
        print(f"Starting batch processing at {datetime.now()}")
        print(f"Input file: {input_file}")
        print(f"Output file: {output_file}")
        print(f"Calculator: {self.calculator.name}")
        print(f"Method: {self.method.name}")
        print("="*60)
        
        # Load input data
        try:
            input_df = pd.read_csv(input_file)
            
            # Find SMILES column
            smiles_col = None
            for col in ['smiles', 'SMILES', 'Smiles']:
                if col in input_df.columns:
                    smiles_col = col
                    break
            
            if smiles_col is None:
                raise ValueError("No SMILES column found in input CSV")
            
            smiles_list = input_df[smiles_col].tolist()
            
            # Get charges and spins if provided
            charges = input_df[charge_col].tolist() if charge_col and charge_col in input_df.columns else [0] * len(smiles_list)
            spins = input_df[spin_col].tolist() if spin_col and spin_col in input_df.columns else [1] * len(smiles_list)
            
            if max_molecules:
                smiles_list = smiles_list[:max_molecules]
                charges = charges[:max_molecules]
                spins = spins[:max_molecules]
            
            print(f"Processing {len(smiles_list)} molecules...")
            
        except Exception as e:
            print(f"Error loading input CSV: {e}")
            return pd.DataFrame()
        
        # Process molecules
        results_list = []
        start_time = time.time()
        
        for idx, (smiles, charge, spin) in enumerate(zip(smiles_list, charges, spins)):
            mol_start = time.time()
            
            print(f"\nMolecule {idx+1}/{len(smiles_list)}: {smiles}")
            
            try:
                result = self.process_molecule(smiles, charge, spin)
                results_list.append(result.to_dict())
                
                print(f"  Formula: {result.formula}")
                print(f"  Wall time: {result.wall_time:.2f}s")
                
                if result.beta_vec is not None:
                    print(f"  β: {result.beta_vec:.3e} a.u.")
                
                if result.homo_lumo_gap is not None:
                    print(f"  HOMO-LUMO gap: {result.homo_lumo_gap:.2f} eV")
                
                if result.error:
                    print(f"  Error: {result.error}")
                
            except Exception as e:
                print(f"  Failed with exception: {e}")
                results_list.append({
                    'smiles': smiles,
                    'error': str(e),
                    'wall_time': time.time() - mol_start
                })
            
            # Periodic saving
            if (idx + 1) % save_interval == 0:
                df_temp = pd.DataFrame(results_list)
                df_temp.to_csv(output_file.replace('.csv', '_temp.csv'), index=False)
                print(f"  Saved temporary results ({idx+1} molecules)")
        
        # Create final DataFrame
        results_df = pd.DataFrame(results_list)
        
        # Add summary statistics
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Batch processing completed!")
        print(f"Total molecules: {len(results_list)}")
        print(f"Total wall time: {total_time:.2f}s")
        print(f"Average time per molecule: {total_time/len(results_list):.2f}s")
        
        # Save final results
        results_df.to_csv(output_file, index=False)
        print(f"Results saved to: {output_file}")
        
        # Print summary statistics
        self.print_summary_statistics(results_df)
        
        return results_df
    
    def print_summary_statistics(self, df: pd.DataFrame):
        """Print summary statistics of the batch run."""
        print(f"\n{'='*60}")
        print("Summary Statistics:")
        print(f"{'='*60}")
        
        # Count successful calculations
        successful = df['beta_vec'].notna().sum() if 'beta_vec' in df.columns else 0
        print(f"Successful calculations: {successful}/{len(df)}")
        
        if successful > 0 and 'beta_vec' in df.columns:
            valid_beta = df['beta_vec'].dropna()
            
            print(f"\nβ statistics:")
            print(f"  Mean: {valid_beta.mean():.3e} a.u.")
            print(f"  Median: {valid_beta.median():.3e} a.u.")
            print(f"  Max: {valid_beta.max():.3e} a.u.")
            print(f"  Min: {valid_beta.min():.3e} a.u.")
            print(f"  Std: {valid_beta.std():.3e} a.u.")
            
            # HOMO-LUMO gap statistics if available
            if 'homo_lumo_gap' in df.columns:
                valid_gap = df['homo_lumo_gap'].dropna()
                if len(valid_gap) > 0:
                    print(f"\nHOMO-LUMO gap statistics:")
                    print(f"  Mean: {valid_gap.mean():.2f} eV")
                    print(f"  Min: {valid_gap.min():.2f} eV")
                    print(f"  Max: {valid_gap.max():.2f} eV")
            
            # Top 5 molecules
            if 'formula' in df.columns:
                top_5 = df.nlargest(5, 'beta_vec')[['smiles', 'beta_vec', 'formula']]
                print(f"\nTop 5 molecules by β:")
                for i, row in top_5.iterrows():
                    print(f"  {row['smiles']}: {row['beta_vec']:.3e} a.u. ({row['formula']})")
            else:
                top_5 = df.nlargest(5, 'beta_vec')[['smiles', 'beta_vec']]
                print(f"\nTop 5 molecules by β:")
                for i, row in top_5.iterrows():
                    print(f"  {row['smiles']}: {row['beta_vec']:.3e} a.u.")