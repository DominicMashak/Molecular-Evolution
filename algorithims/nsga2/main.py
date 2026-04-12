#!/usr/bin/env python3
"""
NSGA-II Main Entry Point
Supports multi-objective optimization with flexible objective configuration
"""

import argparse
import logging
import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

logging.basicConfig(level=logging.INFO)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'molev_utils')))

from molecule_generator import MoleculeGenerator
from optimizer import NSGA2Optimizer
from archive import BinnedParetoArchive
from individual import Individual
from performance import PerformancePlotter

def parse_optimize_arg(arg):
    """Parse optimize argument: 'maximize' -> ('max', None), 'target:5.0' -> ('target', 5.0)"""
    if arg.lower() in ['maximize', 'max']:
        return ('max', None)
    elif arg.lower() in ['minimize', 'min']:
        return ('min', None)
    elif arg.startswith('target:') or arg.startswith('t:'):
        try:
            _, value = arg.split(':', 1)
            return ('target', float(value))
        except ValueError:
            raise ValueError(f"Invalid target format: {arg}")
    else:
        raise ValueError(f"Unknown optimize type: {arg}")


def validate_objectives(objectives, optimize_objectives):
    """Validate that objectives and optimization directions are valid"""
    valid_objectives = {
        'beta', 'beta_mean', 'natoms', 'energy', 'dipole',
        'homo_lumo_gap', 'alpha', 'gamma', 'transition_dipole',
        'oscillator_strength',
        # Derived objectives
        'beta_gamma_ratio', 'total_energy_atom_ratio',
        'alpha_range_distance', 'homo_lumo_gap_range_distance',
        # SmartCADD drug-design objectives (primary)
        'qed', 'sa_score', 'docking_score', 'lipinski_violations',
        'admet_pass',
        # SmartCADD descriptors (can be used as objectives)
        'mol_weight', 'logp', 'tpsa',
        # SmartCADD range-distance objectives (recommended for drug design)
        'mol_weight_range_distance', 'logp_range_distance', 'tpsa_range_distance',
    }
    
    for obj in objectives:
        if obj not in valid_objectives:
            logging.warning(f"Unknown objective '{obj}'. Valid objectives: {valid_objectives}")
    
    if len(objectives) != len(optimize_objectives):
        raise ValueError(
            f"Number of objectives ({len(objectives)}) must match "
            f"number of optimization directions ({len(optimize_objectives)})"
        )
    
    return True


def recalculate_from_database(results_dir):
    """
    Recalculate archive, hypervolume, MOQD, and plots from existing all_molecules_database.json.
    
    Args:
        results_dir (str): Path to the results directory containing all_molecules_database.json
    """
    from rdkit import Chem, rdBase
    
    results_path = Path(results_dir)
    db_file = results_path / 'all_molecules_database.json'
    
    if not db_file.exists():
        logging.error(f"Database file not found: {db_file}")
        return
    
    # Load database
    with open(db_file, 'r') as f:
        db = json.load(f)
    
    logging.info(f"Loaded {len(db)} molecules from {db_file}")
    
    # Find max generation
    max_gen = max((m.get('generation', 0) for m in db), default=0)
    logging.info(f"Max generation: {max_gen}")
    
    # Reconstruct metrics for each generation
    metrics = {
        'generation': [],
        'hypervolume': [],
        'moqd': [],
        'max_beta': [],
        'avg_beta': [],
        'min_atoms': [],
        'avg_atoms': [],
        'pareto_size': [],
        'population_diversity': [],
        'qd_bins': []  # Add QD bins metric
    }
    
    optimize_objectives = [('max', None), ('min', None)]
    
    for gen in range(max_gen + 1):
        # Get molecules up to this generation
        molecules_up_to_gen = [m for m in db if m.get('generation', 0) <= gen]
        if not molecules_up_to_gen:
            continue
        
        # Reconstruct individuals
        individuals = []
        for mol in molecules_up_to_gen:
            if 'objectives' in mol and 'smiles' in mol:
                ind = Individual(
                    smiles=mol['smiles'],
                    objectives=mol['objectives'],
                    generation=mol.get('generation', 0)
                )
                # Compute num_atoms and num_bonds using RDKit
                mol_obj = Chem.MolFromSmiles(mol['smiles'])
                num_atoms = mol_obj.GetNumAtoms() if mol_obj else 0
                num_bonds = mol_obj.GetNumBonds() if mol_obj else 0
                
                # Bin using same scheme as MAP-Elites
                num_atoms_bin = min(19, max(0, (num_atoms - 5) // 2))
                num_bonds_bin = min(19, max(0, (num_bonds - 5) // 2))
                
                # Set features for binning (use num_atoms_bin and num_bonds_bin)
                ind.homo_lumo_gap = num_atoms_bin
                ind.transition_dipole = num_bonds_bin
                ind.oscillator_strength = 0.0
                ind.gamma = 0.0
                ind.beta_surrogate = mol.get('beta_surrogate', mol.get('beta', 0.0))
                ind.natoms = num_atoms
                individuals.append(ind)
        
        # Create archive with 20 bins to match MAP-Elites
        archive = BinnedParetoArchive(n_bins=20, max_size=1000, optimize_objectives=optimize_objectives)
        
        # Rebuild archive
        for ind in individuals:
            archive.add(ind)
        
        # Compute metrics
        global_hv = archive.compute_global_hypervolume()
        moqd = archive.compute_moqd_score()
        population = archive.get_all_individuals()
        
        # Beta values
        betas = [getattr(ind, 'beta_surrogate', None) for ind in population]
        betas = [b for b in betas if b is not None]
        max_beta = max(betas) if betas else 0.0
        avg_beta = float(np.mean(betas)) if betas else 0.0
        
        # Atom counts
        atoms = [getattr(ind, 'natoms', None) for ind in population]
        atoms = [a for a in atoms if a is not None]
        min_atoms = min(atoms) if atoms else 0
        avg_atoms = float(np.mean(atoms)) if atoms else 0.0
        
        # Pareto size
        pareto_size = len(population)
        
        # Diversity
        if population and len(population) > 1:
            obj_array = np.array([getattr(ind, 'objectives', []) for ind in population])
            if obj_array.size:
                diversity = float(np.mean(np.std(obj_array, axis=0)))
            else:
                diversity = 0.0
        else:
            diversity = 0.0
        
        # Append to metrics
        metrics['generation'].append(gen)
        metrics['hypervolume'].append(global_hv)
        metrics['moqd'].append(moqd)
        metrics['max_beta'].append(max_beta)
        metrics['avg_beta'].append(avg_beta)
        metrics['min_atoms'].append(min_atoms)
        metrics['avg_atoms'].append(avg_atoms)
        metrics['pareto_size'].append(pareto_size)
        metrics['population_diversity'].append(diversity)
        metrics['qd_bins'].append(archive.size())  # Add QD bins
        
        logging.info(f"Generation {gen}: HV={global_hv:.6f}, MOQD={moqd:.6f}, Pareto size={pareto_size}, QD bins={archive.size()}")
    
    # Save full metrics
    metrics_file = results_path / 'performance_metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    logging.info(f"Saved reconstructed performance metrics to {metrics_file}")
    
    # Now compute final archive from all molecules
    individuals = []
    for mol in db:
        if 'objectives' in mol and 'smiles' in mol:
            ind = Individual(
                smiles=mol['smiles'],
                objectives=mol['objectives'],
                generation=mol.get('generation', 0)
            )
            # Compute num_atoms and num_bonds using RDKit
            mol_obj = Chem.MolFromSmiles(mol['smiles'])
            num_atoms = mol_obj.GetNumAtoms() if mol_obj else 0
            num_bonds = mol_obj.GetNumBonds() if mol_obj else 0
            
            # Bin using same scheme as MAP-Elites
            num_atoms_bin = min(19, max(0, (num_atoms - 5) // 2))
            num_bonds_bin = min(19, max(0, (num_bonds - 5) // 2))
            
            # Set features for binning
            ind.homo_lumo_gap = num_atoms_bin
            ind.transition_dipole = num_bonds_bin
            ind.oscillator_strength = 0.0
            ind.gamma = 0.0
            individuals.append(ind)
    
    # Create final archive with 20 bins
    archive = BinnedParetoArchive(n_bins=20, max_size=1000, optimize_objectives=optimize_objectives)
    
    # Rebuild archive
    for ind in individuals:
        archive.add(ind)
    
    logging.info(f"Final archive rebuilt with {archive.total_individuals()} individuals in {archive.size()} bins")
    
    # Generate archive heatmap
    heatmap = np.full((20, 20), np.nan)
    for bin_coords, front in archive.cells.items():
        if front:
            max_beta = max(ind.objectives[0] for ind in front)  # Assuming first objective is beta
            heatmap[bin_coords[0], bin_coords[1]] = max_beta
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap, cmap='viridis', cbar_kws={'label': 'Max Beta'})
    plt.xlabel('Num Bonds Bin')
    plt.ylabel('Num Atoms Bin')
    plt.title('Archive Heatmap')
    heatmap_file = results_path / 'archive_heatmap.png'
    plt.savefig(heatmap_file, dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved archive heatmap to {heatmap_file}")
    
    # Regenerate plots
    population = archive.get_all_individuals()
    plotter = PerformancePlotter(results_path)
    plotter.objectives = ['beta', 'natoms']
    plotter.optimize_objectives = [('max', None), ('min', None)]
    plotter.plot_pareto_front(population, generation=999)  # Dummy generation for recalculation
    
    # Save the rebuilt archive to a JSON file
    archive_data = {
        'bins': {
            str(bin_coords): [
                {
                    'smiles': ind.smiles,
                    'objectives': ind.objectives,
                    'generation': getattr(ind, 'generation', 0),
                    'homo_lumo_gap': getattr(ind, 'homo_lumo_gap', 0.0),
                    'transition_dipole': getattr(ind, 'transition_dipole', 0.0),
                    'oscillator_strength': getattr(ind, 'oscillator_strength', 0.0),
                    'gamma': getattr(ind, 'gamma', 0.0)
                } for ind in front
            ] for bin_coords, front in archive.cells.items()
        },
        'total_individuals': archive.total_individuals(),
        'size': archive.size()
    }
    archive_file = results_path / 'archive.json'
    with open(archive_file, 'w') as f:
        json.dump(archive_data, f, indent=2)
    logging.info(f"Saved rebuilt archive to {archive_file}")
    
    # Generate convergence plots (HV, MOQD, etc.)
    from performance import load_results, plot_convergence
    results = load_results(str(results_path))
    plot_convergence(results, results_path)
    
    logging.info("Recalculation complete. Check the results directory for updated plots and metrics.")


def main():
    parser = argparse.ArgumentParser(
        description="NSGA-II Multi-Objective Molecular Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Supported Objectives:
  Quantum Chemistry (--fitness-mode qc):
    beta, beta_mean       - First hyperpolarizability (a.u.)
    natoms                - Number of atoms
    energy                - Total energy (a.u.)
    dipole                - Dipole moment (a.u.)
    homo_lumo_gap         - HOMO-LUMO gap (eV)
    alpha                 - Polarizability (a.u.)
    gamma                 - Second hyperpolarizability (a.u.)
    transition_dipole     - Transition dipole moment (a.u.)
    oscillator_strength   - Oscillator strength

  Drug Design (--fitness-mode smartcadd):
    PRIMARY OBJECTIVES (what you should optimize):
      qed                        - Quantitative Estimate of Drug-likeness (0-1, maximize)
      sa_score                   - Synthetic Accessibility (1=easy, 10=hard, minimize)
      docking_score              - Smina binding affinity (kcal/mol, minimize)
      mol_weight_range_distance  - Distance from optimal MW range [150-500 Da] (minimize) **RECOMMENDED**
      lipinski_violations        - Lipinski Rule of Five violations (0-4, minimize)
      admet_pass                 - ADMET/PAINS filter (1=pass, 0=fail, maximize)

    DESCRIPTORS (can be used as objectives if needed):
      mol_weight                 - Molecular weight (Da, raw value)
      logp                       - LogP lipophilicity (raw value)
      tpsa                       - Topological Polar Surface Area (raw value)
      logp_range_distance        - Distance from optimal LogP [0-5] (minimize, optional)
      tpsa_range_distance        - Distance from optimal TPSA [20-130] (minimize, optional)
      natoms                     - Number of heavy atoms

Examples:
  # 2 objectives: maximize beta, minimize natoms
  python main.py --objectives beta natoms --optimize maximize minimize
  
  # 3 objectives: maximize beta, minimize natoms, maximize HOMO-LUMO gap
  python main.py --objectives beta natoms homo_lumo_gap \\
                 --optimize maximize minimize maximize
  
  # 4 objectives with DFT
  python main.py --calculator dft --functional B3LYP --basis 6-31G \\
                 --objectives beta natoms homo_lumo_gap energy \\
                 --optimize maximize minimize maximize minimize \\
                 --pop_size 20 --n_gen 100
  
  # 5 objectives with target natoms
  python main.py --objectives beta natoms homo_lumo_gap alpha gamma \\
                 --optimize maximize target:15 maximize maximize maximize
  
  # Fast optimization with semiempirical
  python main.py --calculator semiempirical --se-method PM7 \\
                 --objectives beta natoms homo_lumo_gap \\
                 --optimize maximize minimize maximize \\
                 --pop_size 30 --n_gen 200

  # Disable stagnation response
  python main.py --objectives beta natoms --optimize maximize minimize \\
                 --no-stagnation-response

  # SmartCADD drug-design: Fast descriptors mode (RECOMMENDED)
  python main.py --fitness-mode smartcadd \\
                 --objectives qed sa_score mol_weight_range_distance lipinski_violations \\
                 --optimize maximize minimize minimize minimize \\
                 --pop_size 50 --n_gen 100

  # SmartCADD with docking against CDK2 (requires full SmartCADD setup)
  python main.py --fitness-mode smartcadd --smartcadd-mode docking \\
                 --protein-code 1AQ1 \\
                 --objectives docking_score qed sa_score mol_weight_range_distance \\
                 --optimize minimize maximize minimize minimize \\
                 --pop_size 30 --n_gen 50
        """
    )
    
    # Calculator options
    parser.add_argument('--calculator', type=str, required=False, 
                       choices=['dft', 'cc', 'semiempirical', 'xtb'],
                       help='Calculator type')
    parser.add_argument('--basis', type=str, default="6-31G", 
                       help='Basis set (for DFT/CC)')
    parser.add_argument('--functional', type=str, default="B3LYP", 
                       help='Functional (for DFT)')
    parser.add_argument('--method', type=str, default=None, 
                       help='Method (for DFT/CC)')
    parser.add_argument('--se-method', type=str, default="PM7", 
                       help='Semiempirical method (PM6, PM7, AM1, etc.)')
    parser.add_argument('--xtb-method', type=str, default="GFN2-xTB",
                       help='xTB method')
    parser.add_argument('--other', type=str, default=None, 
                       help='Other calculator options')
    
    # Population/generation options
    parser.add_argument('--pop_size', type=int, default=100, 
                       help='Population size')
    parser.add_argument('--n_gen', type=int, default=100, 
                       help='Number of generations')
    parser.add_argument('--n-parents', type=int, default=50, 
                       help='Number of parents to select per generation')
    parser.add_argument('--n-children', type=int, default=50, 
                       help='Number of children to generate per generation')
    
    # Mutation options
    parser.add_argument('--adaptive_mutation', action='store_true', 
                       help='Use adaptive mutation (deprecated, always on)')
    parser.add_argument('--mutation_weights', type=str, default=None, 
                       help='JSON file with mutation weights')
    
    # Output options
    parser.add_argument('--plot_every', type=int, default=10, 
                       help='Plot every n generations')
    parser.add_argument('--output_dir', type=str, default="nsga2_results", 
                       help='Output directory for results')
    
    # Objective options
    parser.add_argument('--objectives', nargs='+', required=False,
                       help='List of objectives (e.g., beta natoms homo_lumo_gap)')
    parser.add_argument('--optimize', nargs='+', required=False,
                       help='Optimization types (e.g., maximize minimize target:15.0)')
    parser.add_argument('--reference-points', nargs='+', type=float, default=None,
                       help='Reference points for hypervolume calculation, one per objective (e.g., 0.0 50.0)')
    parser.add_argument('--problem', type=str, default=None,
                       help='Named problem preset (e.g. nlo_4obj, drug_2obj). '
                            'Fills default objectives, optimize directions, and reference points. '
                            'All explicit CLI flags override preset values.')

    # Stagnation response options
    parser.add_argument('--no-stagnation-response', action='store_true',
                       help='Disable stagnation-adaptive mutation')
    parser.add_argument('--stagnation-threshold', type=int, default=5,
                       help='Generations without improvement to trigger stagnation response')
    parser.add_argument('--stagnation-strategy', type=str, default='hybrid',
                       choices=['hybrid', 'weight', 'atoms'],
                       help='Stagnation response strategy')
    
    # Fitness mode options
    parser.add_argument('--fitness-mode', type=str, default='qc',
                       choices=['qc', 'smartcadd'],
                       help='Fitness evaluation mode: "qc" for quantum chemistry (default), '
                            '"smartcadd" for drug-design evaluation')
    parser.add_argument('--smartcadd-path', type=str, default=None,
                       help='Path to SmartCADD repository (auto-detected if not set)')
    parser.add_argument('--smartcadd-mode', type=str, default='descriptors',
                       choices=['descriptors', 'docking'],
                       help='SmartCADD evaluation mode: "descriptors" (fast, RDKit only) '
                            'or "docking" (full Smina docking pipeline)')
    parser.add_argument('--protein-code', type=str, default=None,
                       help='PDB code for docking target (required for docking mode)')
    parser.add_argument('--protein-path', type=str, default=None,
                       help='Local path to protein PDB file (optional, overrides PDB fetch)')
    parser.add_argument('--alert-collection', type=str, default=None,
                       help='Path to ADMET alert collection CSV')
    parser.add_argument('--atom-set', type=str, default=None,
                       choices=['nlo', 'drug'],
                       help='Atom set: "nlo" for C/N/O (default for qc mode), '
                            '"drug" for C/N/O/S/F/Cl/Br (default for smartcadd mode)')
    parser.add_argument('--encoding', type=str, default='smiles',
                       choices=['smiles', 'selfies'],
                       help='Molecular string encoding used internally (smiles or selfies)')
    parser.add_argument('--crossover-rate', type=float, default=0.0,
                       help='Probability of crossover vs mutation per offspring (0.0 = off, default)')

    # Other options
    parser.add_argument('--debug-mopac', action='store_true',
                       help='Enable debugging output for MOPAC')
    parser.add_argument('--seed', type=int, default=None, 
                       help='Random seed for reproducibility')
    parser.add_argument('--field-strength', type=float, default=0.001,
                       help='Field strength for full_tensor method')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--max-natoms', type=int, default=50,
                       help='Maximum number of atoms allowed')
    
    # Archive options
    parser.add_argument('--archive-bins', type=int, default=10,
                       help='Number of bins per dimension for archive')
    parser.add_argument('--archive-max-size', type=int, default=1000,
                       help='Maximum archive size')
    
    # Recalculation option
    parser.add_argument('--recalculate', type=str,
                       help='Recalculate archive, HV, MOQD, and plots from existing all_molecules_database.json in the specified results directory')
    
    args = parser.parse_args()

    # Resolve --problem preset
    from problem_config import resolve_from_args
    _problem = resolve_from_args(args)
    if _problem is not None:
        if args.objectives is None:
            args.objectives = _problem.objective_keys
        if args.optimize is None:
            args.optimize = _problem.optimize_strings
        # nsga2 uses --reference-points (plural)
        if args.reference_points is None:
            args.reference_points = _problem.reference_point
        if args.measure_bounds is None:
            args.measure_bounds = _problem.measure_bounds_flat

    # Handle recalculation mode
    if args.recalculate:
        recalculate_from_database(args.recalculate)
        return
    
    # Validate required arguments for optimization mode
    if args.fitness_mode == 'qc' and not args.calculator:
        parser.error("--calculator is required for qc fitness mode")
    if args.fitness_mode == 'smartcadd' and args.smartcadd_mode == 'docking':
        if not args.protein_code and not args.protein_path:
            parser.error("--protein-code or --protein-path required for docking mode")
    if not args.objectives:
        parser.error("--objectives is required")
    if not args.optimize:
        parser.error("--optimize is required")
    
    # Parse optimization objectives
    try:
        optimize_objectives = [parse_optimize_arg(opt) for opt in args.optimize]
    except ValueError as e:
        parser.error(str(e))
    
    # Validate objectives and optimization
    try:
        validate_objectives(args.objectives, optimize_objectives)
    except ValueError as e:
        parser.error(str(e))
    
    # Validate reference points
    if args.reference_points:
        if len(args.reference_points) != len(args.objectives):
            parser.error("Number of reference points must match number of objectives")
        reference_points = args.reference_points
    else:
        reference_points = None
    
    # Determine atom set
    if args.atom_set:
        atom_set = args.atom_set
    elif args.fitness_mode == 'smartcadd':
        atom_set = 'drug'
    else:
        atom_set = 'nlo'

    # Log configuration
    logging.info("="*60)
    logging.info("NSGA-II Multi-Objective Optimization")
    logging.info("="*60)
    logging.info(f"Fitness mode: {args.fitness_mode}")
    logging.info(f"Atom set: {atom_set}")
    if args.fitness_mode == 'qc':
        logging.info(f"Calculator: {args.calculator}")
    logging.info(f"Objectives ({len(args.objectives)}): {', '.join(args.objectives)}")
    logging.info(f"Optimization: {', '.join([opt[0] for opt in optimize_objectives])}")
    if reference_points:
        logging.info(f"Reference points: {reference_points}")
    logging.info(f"Population size: {args.pop_size}")
    logging.info(f"Generations: {args.n_gen}")
    logging.info(f"Output directory: {args.output_dir}")
    logging.info("="*60)

    # Set random seeds for reproducibility
    if args.seed is not None:
        import random
        import numpy as np
        from rdkit import Chem, rdBase

        random.seed(args.seed)
        np.random.seed(args.seed)
        rdBase.SeedRandomNumberGenerator(args.seed)
        logging.info(f"Random seed set to: {args.seed}")

    # Load mutation weights
    mutation_weights = None
    if args.mutation_weights:
        import json
        try:
            with open(args.mutation_weights, 'r') as f:
                mutation_weights = json.load(f)
            logging.info(f"Loaded mutation weights from {args.mutation_weights}")
        except Exception as e:
            logging.warning(f"Failed to load mutation weights: {e}")

    # Setup calculator kwargs
    calculator_kwargs = {}
    if args.basis:
        calculator_kwargs['basis'] = args.basis
    if args.functional:
        calculator_kwargs['functional'] = args.functional
    if args.method:
        calculator_kwargs['method'] = args.method
    if args.se_method:
        calculator_kwargs['se_method'] = args.se_method
    if args.xtb_method:
        calculator_kwargs['xtb_method'] = args.xtb_method
    if args.debug_mopac:
        calculator_kwargs['debug_mopac'] = True
    
    # Create generator (uses equal mutation weights by default if mutation_weights is None)
    generator = MoleculeGenerator(seed=args.seed if args.seed is not None else 92,
                                  mutation_weights=mutation_weights,
                                  atom_set=atom_set,
                                  encoding=args.encoding)
    
    # Build SmartCADD kwargs if in smartcadd mode
    smartcadd_kwargs = None
    if args.fitness_mode == 'smartcadd':
        smartcadd_kwargs = {
            'mode': args.smartcadd_mode,
        }
        if args.smartcadd_path:
            smartcadd_kwargs['smartcadd_path'] = args.smartcadd_path
        if args.protein_code:
            smartcadd_kwargs['protein_code'] = args.protein_code
        if args.protein_path:
            smartcadd_kwargs['protein_path'] = args.protein_path
        if args.alert_collection:
            smartcadd_kwargs['alert_collection_path'] = args.alert_collection

    # Create optimizer
    optimizer = NSGA2Optimizer(
        generator=generator,
        calculator_type=args.calculator or "dft",
        calculator_kwargs=calculator_kwargs,
        method=args.method or "full_tensor",
        pop_size=args.pop_size,
        n_gen=args.n_gen,
        plot_every=args.plot_every,
        adaptive_mutation=True,
        output_dir=args.output_dir,
        objectives=args.objectives,
        optimize_objectives=optimize_objectives,
        reference_points=reference_points,
        seed=args.seed,
        n_parents=args.n_parents,
        n_children=args.n_children,
        max_natoms=args.max_natoms,
        archive_bins=args.archive_bins,
        archive_max_size=args.archive_max_size,
        enable_stagnation_response=not args.no_stagnation_response,
        stagnation_threshold=args.stagnation_threshold,
        stagnation_strategy=args.stagnation_strategy,
        verbose=args.verbose,
        field_strength=args.field_strength,
        fitness_mode=args.fitness_mode,
        smartcadd_kwargs=smartcadd_kwargs,
        encoding=args.encoding,
        crossover_rate=args.crossover_rate,
    )
    
    # Run optimization
    try:
        optimizer.run()
        logging.info("\n" + "="*60)
        logging.info("OPTIMIZATION COMPLETED SUCCESSFULLY!")
        logging.info("="*60)
        logging.info(f"Results saved to: {args.output_dir}")
        logging.info("\nGenerated files:")
        logging.info("  - pareto_front_molecules.json")
        logging.info("  - all_molecules_database.json")
        logging.info("  - performance_metrics.json")
        logging.info("  - final_convergence.pdf")
        logging.info("  - pareto_front_gen_*.png")
        
    except KeyboardInterrupt:
        logging.info("\n" + "="*60)
        logging.info("OPTIMIZATION INTERRUPTED BY USER")
        logging.info("="*60)
        logging.info("Saving partial results...")
        optimizer.save_results()
        logging.info(f"Partial results saved to: {args.output_dir}")
        sys.exit(1)
        
    except Exception as e:
        logging.error("\n" + "="*60)
        logging.error("OPTIMIZATION FAILED")
        logging.error("="*60)
        logging.error(f"Error: {e}")
        logging.exception("Traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()