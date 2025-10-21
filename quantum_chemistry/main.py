#!/usr/bin/env python3
"""
Molecular Property Calculator - Main CLI Interface

A unified tool for calculating molecular properties using various
quantum chemical methods and backends.
"""

import sys
import os
import argparse

import pandas as pd
import numpy as np

try:
    from methods.cphf import CPHFMethod
except Exception:
    CPHFMethod = None

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--show-cpu-info', action='store_true', help='Show CPU/thread info and exit')
parser.add_argument('--list-solvents', action='store_true', help='List available solvents and dielectric constants and exit')
args, _ = parser.parse_known_args()
if args.show_cpu_info:
    import multiprocessing
    print(f"Detected CPUs: {multiprocessing.cpu_count()}")
    sys.exit(0)
if args.list_solvents:
    from utils.solvents import SOLVENTS
    print("Available solvents and dielectric constants:")
    for name, eps in SOLVENTS.items():
        print(f"{name.title():<20} : {eps}")
    sys.exit(0)
from datetime import datetime
import pandas as pd

# Add the script directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.processor import Processor
from core.types import (
    CalculatorConfig, MethodConfig, CalculatorType, MethodType,
    DFTFunctional, XTBMethod
)
from utils.molecular import canonicalize_smiles
from calculators.setup import get_system_max_memory, get_system_cpu_info


def create_test_data(filename: str):
    """Create a test CSV file with example molecules."""
    test_molecules = [
        "NC1=CC=C(C=C1)N(=O)=O",  # p-nitroaniline
        "CN(C)C1=CC=C(C=C1)N(=O)=O",  # p-dimethylamino nitrobenzene
        "C1=CC=CC=C1",  # benzene
        "NC1=CC=C(C=C1)C=O",  # p-aminobenzaldehyde
        "CC1=CC=C(C=C1)N",  # p-toluidine
        "O=C(C=CC1=CC=C(C=C1)N(C)C)C2=CC=C(C=C2)N(=O)=O",  # Push-pull stilbene
        "C1=CC=C(C=C1)C=CC2=CC=C(C=C2)N(=O)=O",  # p-nitrostilbene
        "NC1=CC=C(C=C1)C#N",  # p-aminobenzonitrile
        "O=C1C=C(N(C)C)C=CC1=CC=CC2=CC=C(C=C2)N(=O)=O",  # Donor-acceptor polyene
        "C1=CC=C(C=C1)N=NC2=CC=C(C=C2)N(C)C",  # Azobenzene derivative
    ]
    
    df = pd.DataFrame({'smiles': test_molecules})
    df.to_csv(filename, index=False)
    print(f"Created test file: {filename}")
    return filename


def check_dependencies():
    """Check if required dependencies are installed."""
    dependencies = {
        'rdkit': 'RDKit',
        'pandas': 'pandas',
        'numpy': 'numpy',
    }
    
    missing = []
    for module, name in dependencies.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(name)
    
    optional = {}
    try:
        import pyscf
        optional['DFT'] = True
    except ImportError:
        optional['DFT'] = False

    
    if missing:
        print("ERROR: Missing required dependencies:")
        for dep in missing:
            print(f"  - {dep}")
        print("\nInstall with: pip install rdkit pandas numpy")
        return False
    
    print("Available calculators:")
    print(f"  - DFT (PySCF): {'✓' if optional['DFT'] else '✗ (install with: pip install pyscf)'}")
    print()
    
    if not any(optional.values()):
        print("ERROR: No calculator backends available!")
        print("Install at least one of:")
        print("  - pip install pyscf    (for DFT calculations)")
        return False
    
    return True

def setup_calculator_config(args) -> CalculatorConfig:
    """Create calculator configuration from command line arguments."""
    from utils.solvents import get_dielectric

    config = CalculatorConfig(
        calculator_type=CalculatorType[args.calculator.upper()],
        verbose=args.verbose
    )

    if config.calculator_type == CalculatorType.DFT:
        config.functional = args.functional
        config.basis = args.basis
    elif config.calculator_type == CalculatorType.XTB:
        config.xtb_method = args.xtb_method
    elif config.calculator_type.name == "SEMIEMPIRICAL":
        config.se_method = args.se_method

    # Handle solvent and dielectric
    # Accept either a solvent name or a numeric dielectric constant for --solvent
    solvent_input = getattr(args, 'solvent', None)
    dielectric = None
    if solvent_input:
        try:
            # Try to interpret as a float
            dielectric = float(solvent_input)
            config.solvent = None
            config.dielectric = dielectric
        except ValueError:
            # Not a float, treat as solvent name
            config.solvent = solvent_input
            config.dielectric = get_dielectric(solvent_input)
    else:
        config.solvent = "none"
        config.dielectric = get_dielectric("none")

    # Add debug flag - this should only be True if explicitly set via CLI
    # Use False as default to ensure it's only True when --debug-mopac is passed
    config.debug_mopac = getattr(args, 'debug_mopac', False)

    # Add properties to config
    config.properties = getattr(args, 'properties', None)
    if config.properties is None:
        config.properties = ['beta', 'dipole', 'homo_lumo_gap', 'transition_dipole', 'oscillator_strength', 'gamma', 'energy', 'alpha']  # Default to all
    return config


def setup_method_config(args) -> MethodConfig:
    """Create method configuration from command line arguments."""
    # If semiempirical and PM6/PM7, skip method config (MOPAC handles everything)
    if getattr(args, 'calculator', '').lower() == 'semiempirical' and getattr(args, 'se_method', '').upper() in ('PM6', 'PM7'):
        return None
    
    # Map CLI method string to MethodType
    method_str = args.method.upper() if args.method else 'FINITE_FIELD'
    
    try:
        method_type = MethodType[method_str]
    except KeyError:
        available_methods = [m.name.lower() for m in MethodType]
        raise ValueError(
            f"Unknown method '{args.method}'. "
            f"Available methods: {', '.join(available_methods)}"
        )
    
    config = MethodConfig(
        method_type=method_type,
        verbose=args.verbose
    )
    
    if args.field_strength:
        config.field_strength = args.field_strength
    
    return config


def process_single_molecule(args):
    """Process a single SMILES string."""
    calc_config = setup_calculator_config(args)
    method_config = setup_method_config(args)

    # Special handling for MOPAC PM6/PM7: don't pass None for method_config, pass a dummy object
    is_mopac = (
        getattr(args, 'calculator', '').lower() == 'semiempirical'
        and getattr(args, 'se_method', '').upper() in ('PM6', 'PM7')
    )
    if is_mopac:
        from core.types import MethodType
        class DummyMethodConfig:
            method_type = MethodType.FINITE_FIELD  # Placeholder, ignored by Processor for PM6/PM7
            verbose = getattr(args, 'verbose', False)
        method_config = DummyMethodConfig()

    try:
        processor = Processor.from_config(calc_config, method_config, args.verbose)
    except ImportError as e:
        print(f"Error: {e}")
        return

    # Print system info
    max_ram = get_system_max_memory()
    cpu_info = get_system_cpu_info()
    print(f"System RAM allocated: {max_ram} MB")
    print(f"CPU cores: {cpu_info['physical_cores']} physical, {cpu_info['logical_cores']} logical")
    print()

    # Canonicalize SMILES
    canonical_smiles = canonicalize_smiles(args.smiles)
    if canonical_smiles != args.smiles:
        print(f"Canonicalized SMILES: {canonical_smiles}")

    # Print calculator and method from config
    calc_type = calc_config.calculator_type.name
    if calc_type == "DFT":
        print(f"Calculator: DFT/{calc_config.functional}/{calc_config.basis}")
    elif calc_type == "XTB":
        print(f"Calculator: xTB/{calc_config.xtb_method}")
    elif calc_type == "SEMIEMPIRICAL":
        print(f"Calculator: SemiEmpirical/{calc_config.se_method}")
        if getattr(args, 'se_method', '').upper() in ('PM6', 'PM7'):
            print("Method: MOPAC (method selection ignored)")
        else:
            print(f"Method: {method_config.method_type.name}")
    else:
        print(f"Calculator: {calc_type}")
        print(f"Method: {method_config.method_type.name}")
    print("-" * 60)
    
    # Process molecule
    result = processor.process_molecule(canonical_smiles, args.charge, args.spin)
    
    # Print results
    print(f"\n{'='*80}")
    print(f"CALCULATION RESULTS")
    print(f"{'='*80}")
    
    print(f"SMILES:           {result.smiles}")
    print(f"Formula:          {result.formula}")
    if is_mopac:
        print(f"Method:           SemiEmpirical/{getattr(args, 'se_method', '').upper()}/mopac")
    else:
        print(f"Method:           {result.method}")
    print(f"Calculation time: {result.wall_time:.2f} seconds")
    
    if result.error:
        print(f"\nERROR: {result.error}")
        return
    
    # After calculation, format output
    def format_value(val, sci):
        if val is None:
            return "None"
        if sci:
            return f"{val:.6e}"
        else:
            return f"{val:.6f}"

    # Filter output based on requested properties
    requested = set(calc_config.properties)
    
    if 'beta' in requested or 'beta_mean' in requested:
        print(f"\n{'Hyperpolarizability Results (a.u.):':^40}")
        print(f"{'-'*40}")
        if is_mopac:
            print(f"β mean:           {format_value(result.beta_mean, True)}")
            beta_esu = result.beta_mean * 8.641e-33 if result.beta_mean is not None else None
            print(f"\nβ mean (esu):     {format_value(beta_esu, True)}")
        else:
            if result.beta_vec is not None and 'beta' in requested:
                print(f"β vector:         {format_value(result.beta_vec, True)}")
                print(f"β xxx:            {format_value(result.beta_xxx, True)}")
                print(f"β yyy:            {format_value(result.beta_yyy, True)}")
                print(f"β zzz:            {format_value(result.beta_zzz, True)}")
                print(f"β mean:           {format_value(result.beta_mean, True)}")
                beta_esu = result.beta_vec * 8.641e-33
                print(f"\nβ vector (esu):   {format_value(beta_esu, True)}")
    
    print(f"\n{'Other Properties:':^40}")
    print(f"{'-'*40}")
    if not is_mopac:
        if 'dipole' in requested and result.dipole_moment is not None:
            print(f"Dipole moment:    {format_value(result.dipole_moment, False)} a.u.")
            dipole_debye = result.dipole_moment * 2.542
            print(f"                  {format_value(dipole_debye, False)} Debye")
        if 'homo_lumo_gap' in requested and result.homo_lumo_gap is not None:
            print(f"HOMO-LUMO gap:    {format_value(result.homo_lumo_gap, False)} eV")
        if 'energy' in requested and result.total_energy is not None:
            print(f"Total energy:     {format_value(result.total_energy, False)} a.u.")
        if 'alpha' in requested and result.alpha_mean is not None:
            print(f"Alpha mean:       {format_value(result.alpha_mean, True)} a.u.")
        if 'transition_dipole' in requested and hasattr(result, 'transition_dipole') and result.transition_dipole is not None:
            print(f"Transition dipole: {format_value(result.transition_dipole, True)} a.u.")
        if 'oscillator_strength' in requested and hasattr(result, 'oscillator_strength') and result.oscillator_strength is not None:
            print(f"Oscillator strength: {format_value(result.oscillator_strength, True)}")
        if 'gamma' in requested and hasattr(result, 'gamma') and result.gamma is not None:
            print(f"Gamma mean:       {format_value(result.gamma, True)} a.u.")
    
    print(f"\n{'='*80}")


def process_batch_smiles(args):
    """Process SMILES strings from a text file."""
    # Check if input file exists
    if not os.path.exists(args.batch):
        print(f"Error: Batch file '{args.batch}' not found.")
        return
    
    # Read SMILES from text file
    smiles_list = []
    try:
        with open(args.batch, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line and not line.startswith('#'):  # Skip empty lines and comments
                    smiles_list.append(line)
        
        if not smiles_list:
            print(f"Error: No valid SMILES found in '{args.batch}'")
            return
            
        print(f"Found {len(smiles_list)} SMILES strings to process")
        
    except Exception as e:
        print(f"Error reading batch file: {e}")
        return
    
    # Setup configurations
    calc_config = setup_calculator_config(args)
    method_config = setup_method_config(args)
    
    # Special handling for MOPAC PM6/PM7
    is_mopac = (
        getattr(args, 'calculator', '').lower() == 'semiempirical'
        and getattr(args, 'se_method', '').upper() in ('PM6', 'PM7')
    )
    if is_mopac and method_config is None:
        from core.types import MethodType
        class DummyMethodConfig:
            method_type = MethodType.FINITE_FIELD
            verbose = getattr(args, 'verbose', False)
        method_config = DummyMethodConfig()
    
    # Create processor
    try:
        processor = Processor.from_config(calc_config, method_config, args.verbose)
    except ImportError as e:
        print(f"Error: {e}")
        return
    
    # Print system info
    max_ram = get_system_max_memory()
    cpu_info = get_system_cpu_info()
    print(f"System RAM allocated: {max_ram} MB")
    print(f"CPU cores: {cpu_info['physical_cores']} physical, {cpu_info['logical_cores']} logical")
    print()
    
    # Process molecules
    results = []
    failed_count = 0
    
    print(f"\nStarting batch processing...")
    print(f"Output file: {args.output}")
    print(f"Output format: {args.format.upper()}")
    print("-" * 60)
    
    for i, smiles in enumerate(smiles_list, 1):
        try:
            print(f"Processing {i}/{len(smiles_list)}: {smiles}")
            
            # Canonicalize SMILES
            canonical_smiles = canonicalize_smiles(smiles)
            
            # Process molecule with properties
            result = processor.process_molecule(canonical_smiles, args.charge, args.spin)
            
            if result.error:
                print(f"  ERROR: {result.error}")
                failed_count += 1
            else:
                print(f"  Success: β_mean = {result.beta_mean:.6e} a.u.")
            
            # Convert result to dictionary for storage
            result_dict = {'smiles': result.smiles, 'formula': result.formula, 'method': result.method, 'charge': args.charge, 'spin': args.spin, 'wall_time': result.wall_time, 'error': result.error}
            if 'beta' in calc_config.properties or 'beta_mean' in calc_config.properties:
                result_dict.update({'beta_vec': result.beta_vec, 'beta_xxx': result.beta_xxx, 'beta_yyy': result.beta_yyy, 'beta_zzz': result.beta_zzz, 'beta_mean': result.beta_mean})
            if 'dipole' in calc_config.properties:
                result_dict['dipole_moment'] = result.dipole_moment
            if 'homo_lumo_gap' in calc_config.properties:
                result_dict['homo_lumo_gap'] = result.homo_lumo_gap
            if 'energy' in calc_config.properties:
                result_dict['total_energy'] = result.total_energy
            if 'alpha' in calc_config.properties:
                result_dict['alpha_mean'] = result.alpha_mean
            if 'transition_dipole' in calc_config.properties and hasattr(result, 'transition_dipole'):
                result_dict['transition_dipole'] = result.transition_dipole
            if 'oscillator_strength' in calc_config.properties and hasattr(result, 'oscillator_strength'):
                result_dict['oscillator_strength'] = result.oscillator_strength
            if 'gamma' in calc_config.properties and hasattr(result, 'gamma'):
                result_dict['gamma'] = result.gamma
            
            results.append(result_dict)
            
            # Save intermediate results every N molecules
            if i % args.save_interval == 0:
                save_batch_results(results, args.output, args.format)
                print(f"  Saved intermediate results ({i} molecules)")
                
        except Exception as e:
            print(f"  ERROR: Unexpected error: {e}")
            failed_count += 1
            results.append({
                'smiles': smiles,
                'error': str(e),
                'formula': None,
                'method': None,
                'charge': args.charge,
                'spin': args.spin,
                'beta_vec': None,
                'beta_xxx': None,
                'beta_yyy': None,
                'beta_zzz': None,
                'beta_mean': None,
                'dipole_moment': None,
                'homo_lumo_gap': None,
                'total_energy': None,
                'alpha_mean': None,
                'wall_time': None
            })
    
    # Save final results
    save_batch_results(results, args.output, args.format)
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"Total molecules: {len(smiles_list)}")
    print(f"Successful: {len(smiles_list) - failed_count}")
    print(f"Failed: {failed_count}")
    print(f"Output file: {args.output}")
    print(f"{'='*80}")


def save_batch_results(results, output_file, format_type):
    """Save batch results to file in specified format."""
    try:
        if format_type == 'csv':
            df = pd.DataFrame(results)
            df.to_csv(output_file, index=False)
        elif format_type == 'json':
            import json
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    except Exception as e:
        print(f"Error saving results: {e}")


def process_batch(args):
    """Process molecules from CSV file."""
    # Check if input file exists
    if not os.path.exists(args.input):
        if args.create_test:
            args.input = create_test_data(args.input)
        else:
            print(f"Error: Input file '{args.input}' not found.")
            print("Use --create-test to generate a test file.")
            return
    
    # Setup configurations
    calc_config = setup_calculator_config(args)
    method_config = setup_method_config(args)
    
    # Create processor
    try:
        processor = Processor.from_config(calc_config, method_config, args.verbose)
    except ImportError as e:
        print(f"Error: {e}")
        return
    
    # Print system info
    max_ram = get_system_max_memory()
    cpu_info = get_system_cpu_info()
    print(f"System RAM allocated: {max_ram} MB")
    print(f"CPU cores: {cpu_info['physical_cores']} physical, {cpu_info['logical_cores']} logical")
    print()
    
    # Process batch
    results_df = processor.process_batch(
        input_file=args.input,
        output_file=args.output,
        max_molecules=args.max,
        save_interval=args.save_interval,
        charge_col=args.charge_col,
        spin_col=args.spin_col
    )
    

def main():
    # Main entry point for the CLI.
    parser = argparse.ArgumentParser(
        description='Molecular Property Calculator - Unified Interface',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single molecule calculations
  main.py --smiles "NC1=CC=C(C=C1)N(=O)=O"
  main.py --smiles "C-C-O" --calculator dft --functional B3LYP
  main.py --smiles "c1ccccc1" --calculator xtb --method finite_field
  main.py --smiles "CCO" --calculator semiempirical --se-method PM3 --method finite_field
  main.py --smiles "CCO" --calculator semiempirical --se-method PM6 --method finite_field
  main.py --smiles "CCO" --calculator semiempirical --se-method PM7 --method finite_field

  # Batch processing from CSV
  main.py --input molecules.csv --output results.csv
  main.py --input test.csv --calculator dft --functional PBE0 --basis def2-TZVP
  main.py --input data.csv --calculator xtb --method full_tensor
  main.py --input data.csv --calculator semiempirical --se-method AM1 --method empirical

  # Batch processing from text file with SMILES
  main.py --batch smiles.txt --output results.csv
  main.py --batch molecules.txt --output results.json --format json
  main.py --batch input.txt --output results.csv --calculator xtb --analyze

  # Method comparison
  main.py --input molecules.csv --output dft_results.csv --calculator dft
  main.py --input molecules.csv --output xtb_results.csv --calculator xtb
  main.py --input molecules.csv --output semiempirical_results.csv --calculator semiempirical --se-method PM3
  main.py --compare dft_results.csv xtb_results.csv

  # With analysis
  main.py --input molecules.csv --analyze
  main.py --input molecules.csv --reference experimental.csv --analyze
        """
    )
    # General options
    parser.add_argument('--calculator', type=str, choices=['dft', 'xtb', 'cc', 'semiempirical'], default='dft',
                       help='Calculator backend (dft, xtb, cc, semiempirical)')
    parser.add_argument('--smiles', type=str,
                       help='SMILES string for single molecule calculation')
    parser.add_argument('--batch', type=str,
                       help='Text file containing SMILES strings (one per line) for batch processing')
    parser.add_argument('--input', type=str,
                       help='CSV file for batch processing')
    parser.add_argument('--output', '-o', type=str, default='results.csv',
                       help='Output file for batch results')
    parser.add_argument('--format', type=str, choices=['csv', 'json'], default='csv',
                       help='Output format for batch results (csv or json)')
    parser.add_argument('--functional', type=str, default='B3LYP',
                       help='DFT functional (B3LYP, PBE, PBE0, etc.)')
    parser.add_argument('--basis', type=str, default='6-31G',
                       help='Basis set (6-31G, def2-TZVP, cc-pVDZ, etc.)')
    parser.add_argument('--xtb-method', type=str, default='GFN2-xTB',
                       help='xTB method (GFN0-xTB, GFN1-xTB, GFN2-xTB)')
    parser.add_argument('--se-method', type=str, default='PM3',
                       help='Semiempirical method (MINDO, MNDO, AM1, PM3, PM6, PM7)')
    parser.add_argument('--method', type=str, default=None, 
                       choices=['finite_field', 'full_tensor', 'cphf'],
                       help='Method for hyperpolarizability (finite_field, full_tensor, cphf)')
    parser.add_argument('--field-strength', type=float, default=0.001,
                       help='Electric field strength for finite field methods (a.u.)')
    parser.add_argument('--solvent', type=str, default=None,
                       help='Solvent name or dielectric constant (e.g., water, 78.4, none)')
    parser.add_argument('--debug-mopac', action='store_true', 
                        help='Enable debugging output for MOPAC calculations')

    # Molecule options
    parser.add_argument('--charge', type=int, default=0,
                       help='Molecular charge (for single molecule)')
    parser.add_argument('--spin', type=int, default=1,
                       help='Spin multiplicity (2S+1) (for single molecule)')
    parser.add_argument('--charge-col', type=str,
                       help='Column name for charges in batch mode')
    parser.add_argument('--spin-col', type=str,
                       help='Column name for spins in batch mode')

    # Processing options
    parser.add_argument('--max', type=int,
                       help='Maximum number of molecules to process')
    parser.add_argument('--save-interval', type=int, default=10,
                       help='Save interval for batch processing')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')

    # Utility options
    parser.add_argument('--create-test', action='store_true',
                       help='Create test input file if not found')
    parser.add_argument('--check-deps', action='store_true',
                       help='Check dependencies and exit')
    parser.add_argument('--scientific', action='store_true',
                       help='Display numeric results in scientific notation (default: False)')
    parser.add_argument('--properties', nargs='*', default=None,
                       help='List of properties to calculate (beta dipole homo_lumo_gap transition_dipole oscillator_strength gamma energy alpha). Default: all')
    args = parser.parse_args()

    if args.check_deps or len(sys.argv) == 1:
        check_dependencies()
        if args.check_deps:
            return
    
    if args.smiles:
        process_single_molecule(args)
    elif args.batch:
        process_batch_smiles(args)
    elif args.input or args.create_test:
        if not hasattr(args, 'input') or not args.input:
            args.input = 'test_molecules.csv'
        process_batch(args)
    else:
        check_dependencies()

import sys
print("Python executable:", sys.executable)
try:
    import pyscf
    print("PySCF found at:", pyscf.__file__)
except ImportError:
    print("PySCF not found!")

if __name__ == "__main__":
    main()