import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
from pymoo.indicators.hv import Hypervolume
from matplotlib.ticker import FuncFormatter
import re
from rdkit import Chem


def compute_descriptors_from_smiles(smiles: str) -> Tuple[int, int]:
    """
    Compute num_atoms and num_bonds from SMILES string.

    Returns:
        (num_atoms, num_bonds) tuple
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return (0, 0)
        num_atoms = mol.GetNumAtoms()
        num_bonds = mol.GetNumBonds()
        return (num_atoms, num_bonds)
    except:
        return (0, 0)


def populate_missing_descriptors(molecules: List[dict]) -> List[dict]:
    """
    Populate num_atoms and num_bonds fields for molecules that are missing them.

    Args:
        molecules: List of molecule dictionaries

    Returns:
        Updated list of molecules with descriptors populated
    """
    for mol in molecules:
        # Check if descriptors are missing or zero
        if 'num_atoms' not in mol or mol.get('num_atoms', 0) == 0:
            smiles = mol.get('smiles', '')
            if smiles:
                num_atoms, num_bonds = compute_descriptors_from_smiles(smiles)
                mol['num_atoms'] = num_atoms
                mol['num_bonds'] = num_bonds
            else:
                mol['num_atoms'] = 0
                mol['num_bonds'] = 0

        # Also ensure num_bonds exists even if num_atoms exists
        if 'num_bonds' not in mol or mol.get('num_bonds', 0) == 0:
            smiles = mol.get('smiles', '')
            if smiles and mol.get('num_atoms', 0) > 0:
                num_atoms, num_bonds = compute_descriptors_from_smiles(smiles)
                mol['num_bonds'] = num_bonds

    return molecules


def parse_sa_progress_file(file_path: Path) -> List[dict]:
    """Parse SA annealing_progress.txt file."""
    molecules = []
    with open(file_path, 'r') as f:
        current_iteration = None
        for line in f:
            iter_match = re.search(r'Iteration:\s*(\d+)', line)
            if iter_match:
                current_iteration = int(iter_match.group(1))

            match = re.search(r'Best molecule:\s*([^,]+),\s*Beta:\s*([-\d.]+)', line)
            if match and current_iteration is not None:
                smiles = match.group(1).strip()
                fitness_value = float(match.group(2))
                molecule = {
                    'smiles': smiles,
                    'generation': current_iteration,
                    'beta_gamma_ratio': fitness_value,
                    'fitness': fitness_value,
                    'objectives': [fitness_value]  # Single objective
                }
                molecules.append(molecule)
    return molecules


def auto_detect_seeds(results_dir: Path, algorithm_name: str, folder_prefix: str = None) -> List[int]:
    seeds = []

    # Get base algorithm name (before any hyphen variant like -fine, -coarse)
    base_name = algorithm_name.split('-')[0]

    prefix = folder_prefix if folder_prefix else base_name
    prefix_underscore = prefix.replace('_', '_')

    algo_dir = results_dir / algorithm_name
    if algo_dir.exists():
        for item in algo_dir.iterdir():
            if item.is_dir():
                match = re.match(rf'{prefix_underscore}_results_seed_(\d+)', item.name)
                if match:
                    seed_num = int(match.group(1))
                    # For multi-objective and mu+lambda, require JSON database (not just SA progress file)
                    json_file = item / "all_molecules_database.json"
                    if json_file.exists():
                        seeds.append(seed_num)

    return sorted(seeds)


def load_algorithm_data(results_dir: Path, algorithm_name: str, seeds: List[int], folder_prefix: str = None) -> Dict[int, List]:
    """Load molecule data from all seed folders for any algorithm."""
    data_by_seed = {}

    for seed in seeds:
        # Get base algorithm name (before variant suffix like -fine, -coarse)
        base_name = algorithm_name.split('-')[0]

        prefix = folder_prefix if folder_prefix else base_name

        folder_name = f"{prefix}_results_seed_{seed}"
        json_file = results_dir / algorithm_name / folder_name / "all_molecules_database.json"

        if not json_file.exists():
            json_file = results_dir / folder_name / "all_molecules_database.json"

        if json_file.exists():
            with open(json_file, 'r') as f:
                molecules = json.load(f)

            # ALWAYS recompute objectives for ALL algorithms to ensure consistency
            # This handles cases where algorithms may not have saved values correctly
            for m in molecules:
                # Step 1: Recalculate beta_gamma_ratio from beta_mean / gamma
                if 'beta_mean' in m and 'gamma' in m and m['gamma'] != 0:
                    m['beta_gamma_ratio'] = m['beta_mean'] / m['gamma']
                else:
                    m['beta_gamma_ratio'] = m.get('beta_gamma_ratio', 0.0)

                # Step 2: Recalculate total_energy_atom_ratio
                # Try both num_atoms and natoms field names
                natoms = m.get('num_atoms', m.get('natoms', 0))
                if 'total_energy' in m and natoms > 0:
                    m['total_energy_atom_ratio'] = m['total_energy'] / natoms
                else:
                    m['total_energy_atom_ratio'] = m.get('total_energy_atom_ratio', 0.0)

                # Step 3: Calculate alpha_range_distance (target range: [100, 500])
                alpha_mean = m.get('alpha_mean', 0.0)
                if alpha_mean and alpha_mean < 100.0:
                    m['alpha_range_distance'] = 100.0 - alpha_mean
                elif alpha_mean and alpha_mean > 500.0:
                    m['alpha_range_distance'] = alpha_mean - 500.0
                else:
                    m['alpha_range_distance'] = 0.0

                # Step 4: Calculate homo_lumo_gap_range_distance (target range: [2.5, 3.5])
                homo_lumo_gap = m.get('homo_lumo_gap', 0.0)
                if homo_lumo_gap and homo_lumo_gap < 2.5:
                    m['homo_lumo_gap_range_distance'] = 2.5 - homo_lumo_gap
                elif homo_lumo_gap and homo_lumo_gap > 3.5:
                    m['homo_lumo_gap_range_distance'] = homo_lumo_gap - 3.5
                else:
                    m['homo_lumo_gap_range_distance'] = 0.0

                # Step 5: Always set objectives array from recalculated values
                m['objectives'] = [
                    m['beta_gamma_ratio'],
                    m['total_energy_atom_ratio'],
                    m['alpha_range_distance'],
                    m['homo_lumo_gap_range_distance']
                ]

                # Step 6: Ensure generation field exists (default to 0 for algorithms like SA)
                if 'generation' not in m:
                    m['generation'] = 0

            # Populate missing num_atoms and num_bonds from SMILES
            molecules = populate_missing_descriptors(molecules)

            data_by_seed[seed] = molecules
            print(f"  Loaded {len(molecules)} molecules from {algorithm_name} seed {seed}")
        elif 'sa' in algorithm_name or 'simulated_annealing' in algorithm_name:
            base_name = algorithm_name.split('-')[0]
            prefix = folder_prefix if folder_prefix else base_name
            folder_name_sa = f"{prefix}_results_seed_{seed}"

            sa_file = results_dir / algorithm_name / folder_name_sa / "annealing_progress.txt"
            if not sa_file.exists():
                sa_file = results_dir / folder_name_sa / "annealing_progress.txt"

            if sa_file.exists():
                molecules = parse_sa_progress_file(sa_file)
                if molecules:
                    data_by_seed[seed] = molecules
                    print(f"  Loaded {len(molecules)} iterations from SA seed {seed}")
            else:
                print(f"  Warning: File not found: {sa_file}")
        else:
            print(f"  Warning: File not found: {json_file}")

    return data_by_seed


def get_pareto_front(objectives: np.ndarray, maximize_mask: np.ndarray) -> np.ndarray:
    """
    Get Pareto front from objectives array.

    Args:
        objectives: Array of shape (n_solutions, n_objectives)
        maximize_mask: Boolean array indicating which objectives to maximize

    Returns:
        Array of Pareto-optimal objectives
    """
    if len(objectives) == 0:
        return np.array([])

    n_points = len(objectives)
    
    # Convert to maximization problem (higher is better for all)
    obj_transformed = objectives.copy()
    obj_transformed[:, ~maximize_mask] *= -1

    # Find Pareto front using correct dominance definition
    # Point j dominates point i if: j >= i in ALL objectives AND j > i in at least ONE
    is_pareto = np.ones(n_points, dtype=bool)
    
    for i in range(n_points):
        if not is_pareto[i]:
            continue
        for j in range(n_points):
            if i == j or not is_pareto[j]:
                continue
            # Check if j dominates i
            # j dominates i if j >= i everywhere AND j > i somewhere
            if (np.all(obj_transformed[j] >= obj_transformed[i]) and 
                np.any(obj_transformed[j] > obj_transformed[i])):
                is_pareto[i] = False
                break

    return objectives[is_pareto]


def create_grid_archive(molecules: List[dict], grid_bins: List[int],
                       grid_bounds: List[Tuple[float, float]],
                       is_multiobjective: bool,
                       maximize_mask: np.ndarray,
                       n_grid_dims: int = None,
                       descriptor_names: List[str] = None) -> Dict:
    """
    Create MAP-Elites style grid archive from molecules.

    Args:
        molecules: List of molecule dictionaries
        grid_bins: Number of bins per dimension [nbins_dim0, nbins_dim1, ...]
        grid_bounds: Bounds for each dimension [(min0, max0), (min1, max1), ...]
        is_multiobjective: Whether this is multi-objective
        maximize_mask: Which objectives to maximize (for all objectives)
        n_grid_dims: Number of dimensions to use for grid (default: all)
        descriptor_names: Names of descriptor fields to use for binning (e.g., ['num_atoms', 'num_bonds'])

    Returns:
        Dictionary mapping cell indices to lists of molecules
    """
    archive = {}

    # If not specified, use all dimensions for grid
    if n_grid_dims is None:
        n_grid_dims = len(grid_bins)

    # Use descriptors for binning if specified, otherwise use objectives
    use_descriptors = descriptor_names is not None and len(descriptor_names) > 0

    for mol in molecules:
        # Compute cell indices based on descriptors or objectives
        cell_idx = []

        if use_descriptors:
            # Bin based on descriptor values (e.g., num_atoms, num_bonds)
            for dim in range(min(n_grid_dims, len(grid_bins))):
                if dim >= len(descriptor_names):
                    break
                descriptor_name = descriptor_names[dim]
                val = mol.get(descriptor_name, 0)
                n_bins = grid_bins[dim]
                min_val, max_val = grid_bounds[dim]
                # Bin the value
                bin_idx = int((val - min_val) / (max_val - min_val + 1e-10) * n_bins)
                bin_idx = max(0, min(n_bins - 1, bin_idx))  # Clamp to valid range
                cell_idx.append(bin_idx)
        else:
            # Bin based on objectives (original behavior)
            objectives = np.array(mol['objectives'])
            for dim in range(min(n_grid_dims, len(grid_bins))):
                if dim >= len(objectives):
                    break
                n_bins = grid_bins[dim]
                min_val, max_val = grid_bounds[dim]
                val = objectives[dim]
                # Bin the value
                bin_idx = int((val - min_val) / (max_val - min_val + 1e-10) * n_bins)
                bin_idx = max(0, min(n_bins - 1, bin_idx))  # Clamp to valid range
                cell_idx.append(bin_idx)

        cell_idx = tuple(cell_idx)

        # Add molecule to cell
        if cell_idx not in archive:
            archive[cell_idx] = []
        archive[cell_idx].append(mol)

    # For each cell, keep only Pareto front (or best for single-objective)
    if is_multiobjective:
        for cell_idx in archive:
            cell_objs = np.array([m['objectives'] for m in archive[cell_idx]])
            # Get Pareto front considering ALL objectives
            pareto_front = get_pareto_front(cell_objs, maximize_mask)
            # Keep only molecules that are on the Pareto front
            pareto_mols = []
            for mol in archive[cell_idx]:
                mol_obj = np.array(mol['objectives'])
                if any(np.allclose(mol_obj, pf) for pf in pareto_front):
                    pareto_mols.append(mol)
            archive[cell_idx] = pareto_mols
    else:
        # Single objective - keep best in each cell
        for cell_idx in archive:
            best_mol = max(archive[cell_idx],
                          key=lambda m: m['objectives'][0] if maximize_mask[0] else -m['objectives'][0])
            archive[cell_idx] = [best_mol]

    return archive


def compute_qd_metrics(
    data_by_seed: Dict[int, List],
    max_generation: int,
    is_multiobjective: bool,
    ref_point: np.ndarray,
    maximize_mask: np.ndarray,
    grid_bins: List[int],
    eval_interval: int = 1,
    descriptor_names: List[str] = None,
    global_ref_for_hv: np.ndarray = None
) -> Tuple[np.ndarray, Dict[int, Dict[str, np.ndarray]], List[Tuple[float, float]]]:
    """
    Compute QD metrics over generations for an algorithm.

    Returns:
    - generations: Array of generation numbers
    - metrics_by_seed: Dict mapping seed to dict of metric arrays:
        - occupied_cells: Number of grid cells with at least one molecule
        - max_fitness: Maximum fitness (or first objective) ever encountered
        - global_hv: Hypervolume of Pareto front from all molecules ever seen
        - moqd_score: Sum of hypervolumes across all occupied cells
    - grid_bounds: Global grid bounds computed from ALL data (used for consistency)
    """
    generations = np.arange(0, max_generation + 1, eval_interval)
    metrics_by_seed = {}

    # Compute grid bounds ONCE using ALL molecules from ALL generations and ALL seeds
    # This ensures monotonicity as molecules don't get reassigned to different cells
    all_objectives = []
    all_descriptors = []
    for molecules in data_by_seed.values():
        for mol in molecules:
            all_objectives.append(mol['objectives'])
            if descriptor_names:
                all_descriptors.append([mol.get(desc, 0) for desc in descriptor_names])

    all_objectives = np.array(all_objectives)
    n_obj_dims = all_objectives.shape[1]  # Actual number of objectives
    n_grid_dims = len(grid_bins)  # Number of dimensions for grid

    print(f"  Detected {n_obj_dims} objectives, using {n_grid_dims}D grid")
    if descriptor_names:
        print(f"  Using descriptors for binning: {descriptor_names}")

    # Adjust maximize_mask if needed
    if len(maximize_mask) < n_obj_dims:
        # Extend maximize_mask to match number of objectives (assuming remaining are maximized)
        maximize_mask_full = np.ones(n_obj_dims, dtype=bool)
        maximize_mask_full[:len(maximize_mask)] = maximize_mask
        maximize_mask = maximize_mask_full

    # Adjust ref_point if needed
    if len(ref_point) < n_obj_dims:
        ref_point_full = np.zeros(n_obj_dims)
        ref_point_full[:len(ref_point)] = ref_point
        ref_point = ref_point_full

    # Use fixed reference point for hypervolume calculations
    # This ensures monotonicity AND fair cross-algorithm comparison
    if global_ref_for_hv is None:
        # Transform for PyMoo (which expects minimization)
        global_ref_for_hv = ref_point.copy()
        global_ref_for_hv[maximize_mask] *= -1  # Negate maximization objectives

    print(f"  Using FIXED reference point for HV: {global_ref_for_hv}")

    # Calculate global grid bounds based on ALL data, these bounds are fixed for all generations
    grid_bounds = []
    if descriptor_names:
        # Use descriptor bounds for grid
        all_descriptors = np.array(all_descriptors)
        for dim in range(n_grid_dims):
            descriptor_name = descriptor_names[dim] if dim < len(descriptor_names) else None

            # Use fixed bounds for num_atoms (molecular constraint: 5-30 atoms)
            if descriptor_name == 'num_atoms':
                min_val = 5
                max_val = 30
            elif descriptor_name == 'num_bonds':
                # For num_bonds, use integer bounds from data (bonds are discrete)
                min_val = int(np.floor(np.min(all_descriptors[:, dim])))
                max_val = int(np.ceil(np.max(all_descriptors[:, dim])))
            else:
                # For other continuous descriptors, compute from data with padding
                min_val = np.min(all_descriptors[:, dim])
                max_val = np.max(all_descriptors[:, dim])
                # Add 5% padding
                padding = (max_val - min_val) * 0.05
                min_val -= padding
                max_val += padding

            grid_bounds.append((min_val, max_val))
        print(f"  Global grid bounds (descriptors): {grid_bounds}")
    else:
        # Use objective bounds for grid
        for dim in range(n_grid_dims):
            min_val = np.min(all_objectives[:, dim])
            max_val = np.max(all_objectives[:, dim])
            # Add 5% padding
            padding = (max_val - min_val) * 0.05
            grid_bounds.append((min_val - padding, max_val + padding))
        print(f"  Global grid bounds (objectives): {grid_bounds}")

    # Instantiate HV indicator once outside the loop (used for both global HV and MOQD)
    global_hv_indicator = Hypervolume(ref_point=global_ref_for_hv)

    # Compute metrics for each seed using FIXED grid bounds
    for seed, molecules in data_by_seed.items():
        occupied_cells_list = []
        # Track all 4 objectives separately
        obj_0_list = []  # beta_gamma_ratio
        obj_1_list = []  # total_energy_atom_ratio
        obj_2_list = []  # alpha_range_distance
        obj_3_list = []  # homo_lumo_gap_range_distance
        global_hv_list = []
        moqd_score_list = []

        for gen in generations:
            # Get all molecules up to this generation (cumulative)
            gen_molecules = [m for m in molecules if m['generation'] <= gen]

            if not gen_molecules:
                occupied_cells_list.append(0)
                obj_0_list.append(0)
                obj_1_list.append(0)
                obj_2_list.append(0)
                obj_3_list.append(0)
                global_hv_list.append(0)
                moqd_score_list.append(0)
                continue

            # Create archive for this generation using fixed global grid bounds
            # This ensures molecules stay in the same cells across generations (for monotonicity)
            archive = create_grid_archive(gen_molecules, grid_bins, grid_bounds,
                                         is_multiobjective, maximize_mask, n_grid_dims,
                                         descriptor_names=descriptor_names)

            # Metric 1: Number of occupied cells
            n_occupied = len(archive)
            occupied_cells_list.append(n_occupied)

            # Metrics 2-5: Track all 4 objectives separately
            all_objectives_gen = np.array([m['objectives'] for m in gen_molecules])

            # Obj 0: beta_gamma_ratio (maximize)
            obj_0_val = np.max(all_objectives_gen[:, 0]) if maximize_mask[0] else np.min(all_objectives_gen[:, 0])
            obj_0_list.append(obj_0_val)

            # Obj 1: total_energy_atom_ratio (maximize)
            obj_1_val = np.max(all_objectives_gen[:, 1]) if maximize_mask[1] else np.min(all_objectives_gen[:, 1])
            obj_1_list.append(obj_1_val)

            # Obj 2: alpha_range_distance (minimize - lower is better)
            obj_2_val = np.min(all_objectives_gen[:, 2]) if not maximize_mask[2] else np.max(all_objectives_gen[:, 2])
            obj_2_list.append(obj_2_val)

            # Obj 3: homo_lumo_gap_range_distance (minimize - lower is better)
            obj_3_val = np.min(all_objectives_gen[:, 3]) if not maximize_mask[3] else np.max(all_objectives_gen[:, 3])
            obj_3_list.append(obj_3_val)

            # Metric 3: Global hypervolume from ALL molecules ever seen
            all_objectives_gen = np.array([m['objectives'] for m in gen_molecules])

            if is_multiobjective:
                # Get Pareto front from all molecules
                pareto_front = get_pareto_front(all_objectives_gen, maximize_mask)

                if len(pareto_front) > 0:
                    # Compute hypervolume using fixed global reference point
                    # PyMoo's Hypervolume expects minimization, so convert maximization objectives
                    pareto_for_hv = pareto_front.copy()
                    pareto_for_hv[:, maximize_mask] *= -1  # Flip maximization to minimization

                    # Use pre-instantiated HV indicator with fixed global reference point
                    global_hv = global_hv_indicator(pareto_for_hv)
                else:
                    global_hv = 0
            else:
                # Single objective: HV = max_fitness - ref_point
                global_hv = max(0, max_fitness - ref_point[0])

            global_hv_list.append(global_hv)

            # Metric 4: MOQD Score - sum of hypervolumes across all occupied cells
            moqd_score = 0
            
            for cell_idx, cell_mols in archive.items():
                cell_objs = np.array([m['objectives'] for m in cell_mols])

                if is_multiobjective:
                    # cell_mols already contains only Pareto-optimal solutions
                    # from create_grid_archive, so no need to recompute Pareto front
                    if len(cell_objs) > 0:
                        # Convert to minimization for PyMoo
                        cell_for_hv = cell_objs.copy()
                        cell_for_hv[:, maximize_mask] *= -1

                        # Use pre-instantiated HV indicator with FIXED global reference point
                        cell_hv = global_hv_indicator(cell_for_hv)
                        moqd_score += cell_hv
                else:
                    # Single objective: cell contributes (fitness - ref_point)
                    cell_fitness = cell_objs[0, 0]  # Best (only) fitness in cell
                    cell_hv = max(0, cell_fitness - ref_point[0])
                    moqd_score += cell_hv

            moqd_score_list.append(moqd_score)

        # Track cumulative function evaluations
        evaluations_list = []
        for gen in generations:
            n_evals = len([m for m in molecules if m['generation'] <= gen])
            evaluations_list.append(n_evals)

        metrics_by_seed[seed] = {
            'occupied_cells': np.array(occupied_cells_list),
            'beta_gamma_ratio': np.array(obj_0_list),
            'total_energy_atom_ratio': np.array(obj_1_list),
            'alpha_range_distance': np.array(obj_2_list),
            'homo_lumo_gap_range_distance': np.array(obj_3_list),
            'global_hv': np.array(global_hv_list),
            'moqd_score': np.array(moqd_score_list),
            'evaluations': np.array(evaluations_list)
        }

        print(f"    Seed {seed} - Gen {max_generation} ({evaluations_list[-1]} evals): "
              f"Cells={occupied_cells_list[-1]}, "
              f"BetaGamma={obj_0_list[-1]:.2e}, "
              f"EnergyAtom={obj_1_list[-1]:.2e}, "
              f"AlphaDist={obj_2_list[-1]:.2e}, "
              f"HLGapDist={obj_3_list[-1]:.2e}, "
              f"GlobalHV={global_hv_list[-1]:.2e}, "
              f"MOQD={moqd_score_list[-1]:.2e}")

    return generations, metrics_by_seed, grid_bounds


def plot_qd_metrics(
    generations: np.ndarray,
    metrics_by_seed: Dict[int, Dict[str, np.ndarray]],
    algorithm_name: str,
    is_multiobjective: bool,
    output_dir: Path,
    x_axis: str = 'generations'  # 'generations' or 'evaluations'
):
    """Plot QD metrics for a single algorithm with all seeds shown."""
    seeds = sorted(metrics_by_seed.keys())

    # Determine x-axis data
    if x_axis == 'evaluations':
        # Use evaluations for x-axis
        x_data_by_seed = {s: metrics_by_seed[s]['evaluations'] for s in seeds}
        # Compute mean x-axis for aggregate statistics
        x_data_mean = np.mean(np.vstack([x_data_by_seed[s] for s in seeds]), axis=0)
        x_label = 'Function Evaluations'
    else:
        # Use generations for x-axis
        x_data_by_seed = {s: generations for s in seeds}
        x_data_mean = generations
        x_label = 'Generations'

    # Stack metrics
    occupied_matrix = np.vstack([metrics_by_seed[s]['occupied_cells'] for s in seeds])
    beta_gamma_matrix = np.vstack([metrics_by_seed[s]['beta_gamma_ratio'] for s in seeds])
    global_hv_matrix = np.vstack([metrics_by_seed[s]['global_hv'] for s in seeds])
    moqd_matrix = np.vstack([metrics_by_seed[s]['moqd_score'] for s in seeds])

    occupied_mean = np.mean(occupied_matrix, axis=0)
    occupied_median = np.median(occupied_matrix, axis=0)
    occupied_q25 = np.percentile(occupied_matrix, 25, axis=0)
    occupied_q75 = np.percentile(occupied_matrix, 75, axis=0)

    beta_gamma_mean = np.mean(beta_gamma_matrix, axis=0)
    beta_gamma_median = np.median(beta_gamma_matrix, axis=0)
    beta_gamma_q25 = np.percentile(beta_gamma_matrix, 25, axis=0)
    beta_gamma_q75 = np.percentile(beta_gamma_matrix, 75, axis=0)

    global_hv_mean = np.mean(global_hv_matrix, axis=0)
    global_hv_median = np.median(global_hv_matrix, axis=0)
    global_hv_q25 = np.percentile(global_hv_matrix, 25, axis=0)
    global_hv_q75 = np.percentile(global_hv_matrix, 75, axis=0)

    moqd_mean = np.mean(moqd_matrix, axis=0)
    moqd_median = np.median(moqd_matrix, axis=0)
    moqd_q25 = np.percentile(moqd_matrix, 25, axis=0)
    moqd_q75 = np.percentile(moqd_matrix, 75, axis=0)

    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    title_name = algorithm_name.upper().replace('_', '-')

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#96CEB4', '#FFEAA7', '#DDA0DD']

    # Plot 1: Occupied Cells
    ax = axes[0, 0]
    for i, seed in enumerate(seeds):
        ax.plot(x_data_by_seed[seed], metrics_by_seed[seed]['occupied_cells'],
               alpha=0.6, linewidth=1.5, color=colors[i % len(colors)],
               label=f'Seed {seed}', drawstyle='steps-post')
    ax.plot(x_data_mean, occupied_mean, linewidth=3, color='black',
           linestyle='--', alpha=0.8, label='Mean', drawstyle='steps-post')
    ax.plot(x_data_mean, occupied_median, linewidth=3, color='darkgreen',
           linestyle=':', alpha=0.8, label='Median', drawstyle='steps-post')
    ax.fill_between(x_data_mean, occupied_q25, occupied_q75, alpha=0.2,
                    color='gray', label='IQR', step='post')
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel('Occupied Archive Cells', fontsize=11)
    ax.set_title(f'{title_name}: Occupied Cells', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)

    # Plot 2: Beta/Gamma Ratio
    ax = axes[0, 1]
    for i, seed in enumerate(seeds):
        ax.plot(x_data_by_seed[seed], metrics_by_seed[seed]['beta_gamma_ratio'],
               alpha=0.6, linewidth=1.5, color=colors[i % len(colors)],
               label=f'Seed {seed}', drawstyle='steps-post')
    ax.plot(x_data_mean, beta_gamma_mean, linewidth=3, color='black',
           linestyle='--', alpha=0.8, label='Mean', drawstyle='steps-post')
    ax.plot(x_data_mean, beta_gamma_median, linewidth=3, color='darkgreen',
           linestyle=':', alpha=0.8, label='Median', drawstyle='steps-post')
    ax.fill_between(x_data_mean, beta_gamma_q25, beta_gamma_q75, alpha=0.2,
                    color='gray', label='IQR', step='post')
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel('Beta/Gamma Ratio (maximize)', fontsize=11)
    ax.set_title(f'{title_name}: Best Beta/Gamma Ratio', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(style='plain', axis='y')

    # Plot 3: Global Hypervolume
    ax = axes[1, 0]
    for i, seed in enumerate(seeds):
        ax.plot(x_data_by_seed[seed], metrics_by_seed[seed]['global_hv'],
               alpha=0.6, linewidth=1.5, color=colors[i % len(colors)],
               label=f'Seed {seed}', drawstyle='steps-post')
    ax.plot(x_data_mean, global_hv_mean, linewidth=3, color='black',
           linestyle='--', alpha=0.8, label='Mean', drawstyle='steps-post')
    ax.plot(x_data_mean, global_hv_median, linewidth=3, color='darkgreen',
           linestyle=':', alpha=0.8, label='Median', drawstyle='steps-post')
    ax.fill_between(x_data_mean, global_hv_q25, global_hv_q75, alpha=0.2,
                    color='gray', label='IQR', step='post')
    ax.set_xlabel(x_label, fontsize=11)
    if is_multiobjective:
        ax.set_ylabel(r'Global Hypervolume ($\times$10$^8$)', fontsize=11)
        # Use Q75 (upper IQR) with small margin instead of max
        max_q75 = np.max(global_hv_q75)
        y_max_scaled = np.ceil(max_q75 * 1.1 / 1e8)  # 10% headroom above Q75
        ax.set_ylim(0, y_max_scaled * 1e8)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f'{v/1e8:.1f}'))
    else:
        ax.set_ylabel('Hypervolume (Beta/Gamma Ratio)', fontsize=11)
        ax.ticklabel_format(style='plain', axis='y')
    ax.set_title(f'{title_name}: Global Hypervolume', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)

    # Plot 4: MOQD Score
    ax = axes[1, 1]
    for i, seed in enumerate(seeds):
        ax.plot(x_data_by_seed[seed], metrics_by_seed[seed]['moqd_score'],
               alpha=0.6, linewidth=1.5, color=colors[i % len(colors)],
               label=f'Seed {seed}', drawstyle='steps-post')
    ax.plot(x_data_mean, moqd_mean, linewidth=3, color='black',
           linestyle='--', alpha=0.8, label='Mean', drawstyle='steps-post')
    ax.plot(x_data_mean, moqd_median, linewidth=3, color='darkgreen',
           linestyle=':', alpha=0.8, label='Median', drawstyle='steps-post')
    ax.fill_between(x_data_mean, moqd_q25, moqd_q75, alpha=0.2,
                    color='gray', label='IQR', step='post')
    ax.set_xlabel(x_label, fontsize=11)
    if is_multiobjective:
        ax.set_ylabel(r'MOQD Score ($\times$10$^8$)', fontsize=11)
        max_moqd = np.max(moqd_matrix)
        y_max_scaled = np.ceil(max_moqd / 1e8)
        ax.set_ylim(0, y_max_scaled * 1e8)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f'{v/1e8:.1f}'))
    else:
        ax.set_ylabel('MOQD Score', fontsize=11)
        ax.ticklabel_format(style='plain', axis='y')
    ax.set_title(f'{title_name}: MOQD Score', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    suffix = '_evals' if x_axis == 'evaluations' else ''
    output_file = output_dir / f"{algorithm_name}_qd_metrics{suffix}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Plot saved to {output_file}")
    plt.close()


def plot_comparison(
    algo_data: Dict[str, Tuple[np.ndarray, Dict]],
    metric_name: str,
    is_multiobjective: bool,
    output_file: Path,
    x_axis: str = 'generations'
):
    """
    Plot comparison of a single metric across algorithms.
    Shows only median and IQR for clarity.

    Args:
        x_axis: 'generations' or 'evaluations' for x-axis data
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    colors = {
        'sa': '#E74C3C',
        'map_elites': '#3498DB',
        'map_elites-fine': '#3498DB',
        'map_elites-coarse': '#17A2B8',
        'mu_lambda': '#2ECC71',
        'nsga2': '#9B59B6',
        'mome': '#F39C12',
        'mome-fine': '#F39C12',
        'mome-coarse': '#8B4513'
    }

    for algo_name, (generations, metrics_by_seed) in algo_data.items():
        seeds = sorted(metrics_by_seed.keys())
        metric_matrix = np.vstack([metrics_by_seed[s][metric_name] for s in seeds])

        # Determine x-axis data
        if x_axis == 'evaluations':
            x_matrix = np.vstack([metrics_by_seed[s]['evaluations'] for s in seeds])
            x_data = np.median(x_matrix, axis=0)
            x_label = 'Function Evaluations'
        else:
            x_data = generations
            x_label = 'Generations'

        median = np.median(metric_matrix, axis=0)
        q25 = np.percentile(metric_matrix, 25, axis=0)
        q75 = np.percentile(metric_matrix, 75, axis=0)

        label = algo_name.upper().replace('_', '-')
        color = colors.get(algo_name, '#95A5A6')

        ax.plot(x_data, median, linewidth=3, color=color,
               label=f'{label} (Median)', linestyle='-', alpha=0.9, drawstyle='steps-post')
        ax.fill_between(x_data, q25, q75, alpha=0.25,
                       color=color, step='post')

    ax.set_xlabel(x_label, fontsize=13)

    # Set ylabel and formatting based on metric
    if metric_name == 'occupied_cells':
        ax.set_ylabel('Occupied Archive Cells', fontsize=13)
        ax.set_title('Occupied Cells', fontsize=15, fontweight='bold')

    elif metric_name == 'beta_gamma_ratio':
        ax.set_ylabel('Beta/Gamma Ratio (maximize)', fontsize=13)
        ax.set_title('Best Beta/Gamma Ratio', fontsize=15, fontweight='bold')
        ax.ticklabel_format(style='plain', axis='y')

    elif metric_name == 'total_energy_atom_ratio':
        ax.set_ylabel('Total Energy per Atom (maximize)', fontsize=13)
        ax.set_title('Best Energy/Atom Ratio', fontsize=15, fontweight='bold')
        ax.ticklabel_format(style='plain', axis='y')

    elif metric_name == 'alpha_range_distance':
        ax.set_ylabel('Distance from Alpha Target Range (minimize)', fontsize=13)
        ax.set_title('Alpha Range Distance', fontsize=15, fontweight='bold')
        ax.ticklabel_format(style='plain', axis='y')
        ax.invert_yaxis()  # Invert so lower (better) values appear higher

    elif metric_name == 'homo_lumo_gap_range_distance':
        ax.set_ylabel('Distance from HOMO-LUMO Gap Target (minimize)', fontsize=13)
        ax.set_title('HOMO-LUMO Gap Range Distance', fontsize=15, fontweight='bold')
        ax.ticklabel_format(style='plain', axis='y')
        ax.invert_yaxis()  # Invert so lower (better) values appear higher
    elif metric_name == 'global_hv':
        if is_multiobjective:
            ax.set_ylabel(r'Global Hypervolume ($\times$10$^8$)', fontsize=13)
            # Use 95th percentile to avoid outliers compressing the visualization
            all_values = []
            for _, (_, metrics_by_seed) in algo_data.items():
                metric_matrix = np.vstack([metrics_by_seed[s][metric_name] for s in metrics_by_seed.keys()])
                median = np.median(metric_matrix, axis=0)
                all_values.extend(median)
            p95 = np.percentile(all_values, 95)
            y_max_scaled = np.ceil(p95 * 1.2 / 1e8)  # 20% headroom above 95th percentile
            if y_max_scaled > 0:
                ax.set_ylim(0, y_max_scaled * 1e8)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f'{v/1e8:.1f}'))
        else:
            ax.set_ylabel('Hypervolume (Beta/Gamma Ratio)', fontsize=13)
            ax.ticklabel_format(style='plain', axis='y')
        ax.set_title('Global Hypervolume', fontsize=15, fontweight='bold')
    elif metric_name == 'moqd_score':
        if is_multiobjective:
            ax.set_ylabel(r'MOQD Score ($\times$10$^8$)', fontsize=13)
            # Use 95th percentile to avoid outliers compressing the visualization
            all_values = []
            for _, (_, metrics_by_seed) in algo_data.items():
                metric_matrix = np.vstack([metrics_by_seed[s][metric_name] for s in metrics_by_seed.keys()])
                median = np.median(metric_matrix, axis=0)
                all_values.extend(median)
            p95 = np.percentile(all_values, 95)
            y_max_scaled = np.ceil(p95 * 1.2 / 1e8)  # 20% headroom above 95th percentile
            if y_max_scaled > 0:
                ax.set_ylim(0, y_max_scaled * 1e8)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f'{v/1e8:.1f}'))
        else:
            ax.set_ylabel('MOQD Score', fontsize=13)
            ax.ticklabel_format(style='plain', axis='y')
        ax.set_title('MOQD Score', fontsize=15, fontweight='bold')

    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Comparison plot saved to {output_file}")
    plt.close()


def plot_comparison_mean_median(
    algo_data: Dict[str, Tuple[np.ndarray, Dict]],
    metric_name: str,
    is_multiobjective: bool,
    output_file: Path,
    x_axis: str = 'generations'
):
    """
    Plot comparison of a single metric across algorithms.
    Shows only mean (dashed) and median (dotted) lines, no IQR.

    Args:
        x_axis: 'generations' or 'evaluations' for x-axis data
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    colors = {
        'sa': '#E74C3C',
        'map_elites': '#3498DB',
        'map_elites-fine': '#3498DB',
        'map_elites-coarse': '#17A2B8',
        'mu_lambda': '#2ECC71',
        'nsga2': '#9B59B6',
        'mome': '#F39C12',
        'mome-fine': '#F39C12',
        'mome-coarse': '#8B4513'
    }

    for algo_name, (generations, metrics_by_seed) in algo_data.items():
        seeds = sorted(metrics_by_seed.keys())
        metric_matrix = np.vstack([metrics_by_seed[s][metric_name] for s in seeds])

        # Determine x-axis data
        if x_axis == 'evaluations':
            x_matrix = np.vstack([metrics_by_seed[s]['evaluations'] for s in seeds])
            x_data = np.median(x_matrix, axis=0)
            x_label = 'Function Evaluations'
        else:
            x_data = generations
            x_label = 'Generations'

        mean = np.mean(metric_matrix, axis=0)
        median = np.median(metric_matrix, axis=0)

        label = algo_name.upper().replace('_', '-')
        color = colors.get(algo_name, '#95A5A6')

        # Plot mean with dashed line
        ax.plot(x_data, mean, linewidth=2.5, color=color,
               label=f'{label} (Mean)', linestyle='--', alpha=0.9, drawstyle='steps-post')
        # Plot median with dotted line
        ax.plot(x_data, median, linewidth=2.5, color=color,
               label=f'{label} (Median)', linestyle=':', alpha=0.9, drawstyle='steps-post')

    ax.set_xlabel(x_label, fontsize=13)

    # Set ylabel and formatting based on metric
    if metric_name == 'occupied_cells':
        ax.set_ylabel('Occupied Archive Cells', fontsize=13)
        ax.set_title('Occupied Cells', fontsize=15, fontweight='bold')

    elif metric_name == 'beta_gamma_ratio':
        ax.set_ylabel('Beta/Gamma Ratio (maximize)', fontsize=13)
        ax.set_title('Best Beta/Gamma Ratio', fontsize=15, fontweight='bold')
        ax.ticklabel_format(style='plain', axis='y')

    elif metric_name == 'total_energy_atom_ratio':
        ax.set_ylabel('Total Energy per Atom (maximize)', fontsize=13)
        ax.set_title('Best Energy/Atom Ratio', fontsize=15, fontweight='bold')
        ax.ticklabel_format(style='plain', axis='y')

    elif metric_name == 'alpha_range_distance':
        ax.set_ylabel('Distance from Alpha Target Range (minimize)', fontsize=13)
        ax.set_title('Alpha Range Distance', fontsize=15, fontweight='bold')
        ax.ticklabel_format(style='plain', axis='y')
        ax.invert_yaxis()  # Invert so lower (better) values appear higher

    elif metric_name == 'homo_lumo_gap_range_distance':
        ax.set_ylabel('Distance from HOMO-LUMO Gap Target (minimize)', fontsize=13)
        ax.set_title('HOMO-LUMO Gap Range Distance', fontsize=15, fontweight='bold')
        ax.ticklabel_format(style='plain', axis='y')
        ax.invert_yaxis()  # Invert so lower (better) values appear higher
    elif metric_name == 'global_hv':
        if is_multiobjective:
            ax.set_ylabel(r'Global Hypervolume ($\times$10$^8$)', fontsize=13)
            # Use 95th percentile to avoid outliers compressing the visualization
            all_values = []
            for _, (_, metrics_by_seed) in algo_data.items():
                metric_matrix = np.vstack([metrics_by_seed[s][metric_name] for s in metrics_by_seed.keys()])
                mean = np.mean(metric_matrix, axis=0)
                median = np.median(metric_matrix, axis=0)
                all_values.extend(mean)
                all_values.extend(median)
            p95 = np.percentile(all_values, 95)
            y_max_scaled = np.ceil(p95 * 1.2 / 1e8)  # 20% headroom above 95th percentile
            if y_max_scaled > 0:
                ax.set_ylim(0, y_max_scaled * 1e8)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f'{v/1e8:.1f}'))
        else:
            ax.set_ylabel('Hypervolume (Beta/Gamma Ratio)', fontsize=13)
            ax.ticklabel_format(style='plain', axis='y')
        ax.set_title('Global Hypervolume', fontsize=15, fontweight='bold')
    elif metric_name == 'moqd_score':
        if is_multiobjective:
            ax.set_ylabel(r'MOQD Score ($\times$10$^8$)', fontsize=13)
            # Use 95th percentile to avoid outliers compressing the visualization
            all_values = []
            for _, (_, metrics_by_seed) in algo_data.items():
                metric_matrix = np.vstack([metrics_by_seed[s][metric_name] for s in metrics_by_seed.keys()])
                mean = np.mean(metric_matrix, axis=0)
                median = np.median(metric_matrix, axis=0)
                all_values.extend(mean)
                all_values.extend(median)
            p95 = np.percentile(all_values, 95)
            y_max_scaled = np.ceil(p95 * 1.2 / 1e8)  # 20% headroom above 95th percentile
            if y_max_scaled > 0:
                ax.set_ylim(0, y_max_scaled * 1e8)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f'{v/1e8:.1f}'))
        else:
            ax.set_ylabel('MOQD Score', fontsize=13)
            ax.ticklabel_format(style='plain', axis='y')
        ax.set_title('MOQD Score', fontsize=15, fontweight='bold')

    ax.legend(fontsize=10, loc='best', ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Mean+Median comparison plot saved to {output_file}")
    plt.close()


def generate_archive_heatmaps(
    data_by_seed: Dict[int, List],
    algorithm_name: str,
    ref_point: np.ndarray,
    maximize_mask: np.ndarray,
    grid_bins: List[int],
    grid_bounds: List[Tuple[float, float]],
    output_dir: Path,
    descriptor_names: List[str] = None
):
    """
    Generate heatmap visualizations of the final cumulative archive for each seed.

    For each seed, creates a 2D heatmap where:
    - X-axis: first grid dimension (descriptor or objective)
    - Y-axis: second grid dimension (descriptor or objective)
    - Color intensity: hypervolume of solutions in each cell

    Args:
        grid_bounds: Global grid bounds computed from ALL data (ensures consistency)
    """
    print(f"  Generating archive heatmaps...")

    # Get final generation for each seed and build cumulative archives
    final_archives = {}
    for seed, molecules in data_by_seed.items():
        max_gen = max(m['generation'] for m in molecules)
        # Use all molecules up to final generation (cumulative), not just final generation
        final_mols = [m for m in molecules if m['generation'] <= max_gen]

        # Build cumulative archive using global grid bounds
        all_objectives = np.array([m['objectives'] for m in final_mols])
        n_obj_dims = all_objectives.shape[1]
        n_grid_dims = len(grid_bins)

        # Build archive using global grid bounds (passed as parameter)
        archive = create_grid_archive(
            molecules=final_mols,
            grid_bins=grid_bins,
            grid_bounds=grid_bounds,
            is_multiobjective=(n_obj_dims > 1),
            maximize_mask=maximize_mask,
            n_grid_dims=n_grid_dims,
            descriptor_names=descriptor_names
        )

        final_archives[seed] = archive

    # Generate heatmap for each seed
    for seed, archive in final_archives.items():
        # Initialize 2D grid for heatmap
        heatmap = np.zeros((grid_bins[1], grid_bins[0]))  # rows x cols

        # Calculate hypervolume for each occupied cell
        hv_indicator = Hypervolume(ref_point=ref_point)

        for cell_idx, cell_mols in archive.items():
            # Get objectives for molecules in this cell
            cell_objs = np.array([m['objectives'] for m in cell_mols])

            # Convert objectives for hypervolume calculation (negate if minimizing)
            cell_objs_hv = cell_objs.copy()
            for i in range(len(maximize_mask)):
                if not maximize_mask[i]:
                    cell_objs_hv[:, i] = -cell_objs_hv[:, i]

            # Calculate hypervolume for this cell
            if len(cell_objs_hv) > 0:
                try:
                    # Adjust reference point (negate for minimization objectives)
                    ref_point_adj = ref_point.copy()
                    for i in range(len(maximize_mask)):
                        if not maximize_mask[i]:
                            ref_point_adj[i] = -ref_point_adj[i]

                    hv = hv_indicator(cell_objs_hv)
                    heatmap[cell_idx[1], cell_idx[0]] = hv
                except:
                    # If HV calculation fails, use 0
                    heatmap[cell_idx[1], cell_idx[0]] = 0

        # Create heatmap visualization
        fig, ax = plt.subplots(figsize=(10, 8))

        # Use a colormap where 0 (empty cells) is white
        cmap = plt.cm.viridis.copy()
        cmap.set_bad(color='lightgray')

        # Mask zero values (empty cells)
        masked_heatmap = np.ma.masked_where(heatmap == 0, heatmap)

        # Create heatmap with actual descriptor bounds (not bin indices)
        im = ax.imshow(masked_heatmap, cmap=cmap, aspect='auto', origin='lower',
                      extent=[grid_bounds[0][0], grid_bounds[0][1],
                             grid_bounds[1][0], grid_bounds[1][1]])

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Hypervolume', fontsize=12)

        # Set axis labels based on descriptors or default
        if descriptor_names and len(descriptor_names) >= 2:
            xlabel = f'{descriptor_names[0].replace("_", " ").title()}'
            ylabel = f'{descriptor_names[1].replace("_", " ").title()}'
        else:
            xlabel = 'Beta/Gamma Ratio (binned)'
            ylabel = 'Total Energy/Atom Ratio (binned)'

        ax.set_xlabel(xlabel, fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_title(f'{algorithm_name.upper()} Archive Heatmap - Seed {seed}\n'
                    f'(Final Generation, {len(archive)} occupied cells)',
                    fontsize=15, fontweight='bold')

        # Add grid bounds as text
        bounds_text = (f'{xlabel} range: [{grid_bounds[0][0]:.1f}, {grid_bounds[0][1]:.1f}]\n'
                      f'{ylabel} range: [{grid_bounds[1][0]:.1f}, {grid_bounds[1][1]:.1f}]')
        ax.text(0.02, 0.98, bounds_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()

        # Save heatmap
        output_file = output_dir / f"{algorithm_name}_archive_heatmap_seed_{seed}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"    Heatmap saved to {output_file.name}")
        plt.close()


def generate_median_archive_heatmap(
    data_by_seed: Dict[int, List],
    algorithm_name: str,
    ref_point: np.ndarray,
    maximize_mask: np.ndarray,
    grid_bins: List[int],
    grid_bounds: List[Tuple[float, float]],
    output_dir: Path,
    descriptor_names: List[str] = None
):
    """
    Generate median hypervolume heatmap across all seeds.

    For each cell in the archive grid, computes the median hypervolume
    across all seeds, providing a robust aggregate view of the archive.

    Args:
        grid_bounds: Global grid bounds computed from ALL data (ensures consistency)
    """
    print(f"  Generating median archive heatmap...")

    # Get final generation and build cumulative archives for all seeds
    all_seed_heatmaps = []

    for seed, molecules in data_by_seed.items():
        max_gen = max(m['generation'] for m in molecules)
        # Use all molecules up to final generation (cumulative), not just final generation
        final_mols = [m for m in molecules if m['generation'] <= max_gen]

        # Build cumulative archive using global grid bounds
        all_objectives = np.array([m['objectives'] for m in final_mols])
        n_obj_dims = all_objectives.shape[1]
        n_grid_dims = len(grid_bins)

        # Build archive using global grid bounds (passed as parameter)
        archive = create_grid_archive(
            molecules=final_mols,
            grid_bins=grid_bins,
            grid_bounds=grid_bounds,
            is_multiobjective=(n_obj_dims > 1),
            maximize_mask=maximize_mask,
            n_grid_dims=n_grid_dims,
            descriptor_names=descriptor_names
        )

        # Initialize 2D grid for heatmap for this seed
        heatmap = np.zeros((grid_bins[1], grid_bins[0]))  # rows x cols

        # Calculate hypervolume for each occupied cell
        for cell_idx, cell_mols in archive.items():
            # Get objectives for molecules in this cell
            cell_objs = np.array([m['objectives'] for m in cell_mols])

            # Get Pareto front
            pareto_front = get_pareto_front(cell_objs, maximize_mask)

            if len(pareto_front) > 0:
                try:
                    # Convert to minimization for PyMoo
                    pareto_for_hv = pareto_front.copy()
                    pareto_for_hv[:, maximize_mask] *= -1

                    # Reference point for minimization
                    ref_for_hv = ref_point.copy()
                    ref_for_hv[maximize_mask] *= -1

                    # Adjust ref to be above all cell values
                    for i in range(len(ref_for_hv)):
                        max_val = np.max(pareto_for_hv[:, i])
                        if ref_for_hv[i] < max_val:
                            ref_for_hv[i] = max_val + abs(max_val) * 0.1 + 1.0

                    hv_indicator = Hypervolume(ref_point=ref_for_hv)
                    hv = hv_indicator(pareto_for_hv)
                    heatmap[cell_idx[1], cell_idx[0]] = hv
                except:
                    # If HV calculation fails, use 0
                    heatmap[cell_idx[1], cell_idx[0]] = 0

        all_seed_heatmaps.append(heatmap)

    # Stack all heatmaps and compute median
    heatmaps_array = np.array(all_seed_heatmaps)  # shape: (n_seeds, rows, cols)
    median_heatmap = np.median(heatmaps_array, axis=0)

    # Also compute count of how many seeds occupied each cell
    occupation_count = np.sum(heatmaps_array > 0, axis=0)

    # Create heatmap visualization
    fig, ax = plt.subplots(figsize=(10, 8))

    # Use a colormap where 0 (empty cells) is white
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color='lightgray')

    # Mask zero values (empty cells)
    masked_heatmap = np.ma.masked_where(median_heatmap == 0, median_heatmap)

    # Create heatmap with actual descriptor bounds (not bin indices)
    im = ax.imshow(masked_heatmap, cmap=cmap, aspect='auto', origin='lower',
                  extent=[grid_bounds[0][0], grid_bounds[0][1],
                         grid_bounds[1][0], grid_bounds[1][1]])

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Median Hypervolume', fontsize=12)

    # Set axis labels based on descriptors or default
    if descriptor_names and len(descriptor_names) >= 2:
        xlabel = f'{descriptor_names[0].replace("_", " ").title()}'
        ylabel = f'{descriptor_names[1].replace("_", " ").title()}'
    else:
        xlabel = 'Beta/Gamma Ratio (binned)'
        ylabel = 'Total Energy/Atom Ratio (binned)'

    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)

    n_seeds = len(data_by_seed)
    total_occupied = np.sum(occupation_count > 0)
    ax.set_title(f'{algorithm_name.upper()} Median Archive Heatmap\n'
                f'(Across {n_seeds} seeds, {total_occupied} cells occupied by ≥1 seed)',
                fontsize=15, fontweight='bold')

    # Add grid bounds as text (using global grid bounds)
    bounds_text = (f'{xlabel} range: [{grid_bounds[0][0]:.1f}, {grid_bounds[0][1]:.1f}]\n'
                  f'{ylabel} range: [{grid_bounds[1][0]:.1f}, {grid_bounds[1][1]:.1f}]')
    ax.text(0.02, 0.98, bounds_text, transform=ax.transAxes,
           fontsize=9, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    # Save heatmap
    output_file = output_dir / f"{algorithm_name}_archive_heatmap_median.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"    Median heatmap saved to {output_file.name}")
    plt.close()


def main():
    results_dir = Path("analysis/results")
    figures_base_dir = Path("analysis/figures")
    figures_base_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect all available algorithms
    # We ignore original binning schemes and apply uniform binning for fair comparison
    algo_configs = {
        'map_elites-fine': {},
        'map_elites-coarse': {},
        'mome-fine': {},
        'mome-coarse': {},
        'nsga2': {},
        'sa': {},
        'mu_lambda': {}
    }

    # Auto-detect available seeds for each algorithm
    print("="*70)
    print("AUTO-DETECTING AVAILABLE DATA")
    print("="*70)

    algos = {}
    for algo_name in algo_configs.keys():
        detected_seeds = auto_detect_seeds(results_dir, algo_name, folder_prefix=None)
        if detected_seeds:
            algos[algo_name] = {
                'seeds': detected_seeds,
                'folder_prefix': None
            }
            print(f"  {algo_name}: {len(detected_seeds)} seeds found {detected_seeds}")
        else:
            print(f"  {algo_name}: No data found, skipping")

    print("="*70)
    print("COMPREHENSIVE QUALITY-DIVERSITY ANALYSIS")
    print("="*70)

    # === STEP 1: Load all algorithms' data ===
    print("\n" + "="*70)
    print("STEP 1: LOADING DATA FOR ALL ALGORITHMS")
    print("="*70)

    all_data_by_algo = {}
    for algo_name, config in algos.items():
        seeds = config['seeds']
        folder_prefix = config.get('folder_prefix', None)
        print(f"\n{algo_name.upper()}:")
        print(f"  Loading data...")
        data_by_seed = load_algorithm_data(results_dir, algo_name, seeds, folder_prefix)

        if not data_by_seed:
            print(f"  No data found for {algo_name}")
            continue

        all_data_by_algo[algo_name] = data_by_seed
        total_molecules = sum(len(mols) for mols in data_by_seed.values())
        print(f"  Loaded {total_molecules} total molecules across {len(data_by_seed)} seeds")

    # === STEP 2: Use FIXED reference point (same as used during optimization) ===
    print("\n" + "="*70)
    print("STEP 2: USING FIXED REFERENCE POINT")
    print("="*70)

    # Use the EXACT reference points from the run scripts
    # These are the same values used during optimization for all algorithms
    ref_point = np.array([0.0, 0.0, 500.0, 100.0])
    maximize_mask = np.array([True, False, False, False])

    # Transform reference point for PyMoo (which expects minimization)
    shared_global_ref = ref_point.copy()
    shared_global_ref[maximize_mask] *= -1  # Negate maximization objectives

    print(f"  Original reference point: {ref_point}")
    print(f"  Transformed for HV calculation (PyMoo): {shared_global_ref}")
    print(f"  This FIXED reference point will be used for ALL algorithms to ensure fair comparison")

    # === STEP 3: Run analysis twice - once with COARSE binning, once with FINE binning ===
    # This ensures fair comparison by applying the same archive binning to ALL algorithms

    binning_schemes = [
        {
            'name': 'COARSE',
            'grid_bins': [10, 10],
            'figures_dir': figures_base_dir / "figures_coarse",
            'description': '10x10 grid (coarse binning)'
        },
        {
            'name': 'FINE',
            'grid_bins': [20, 20],
            'figures_dir': figures_base_dir / "figures_fine",
            'description': '20x20 grid (fine binning)'
        }
    ]

    for scheme in binning_schemes:
        print("\n" + "="*70)
        print(f"STEP 3: COMPUTING QD METRICS - {scheme['name']} BINNING SCHEME")
        print(f"  {scheme['description']}")
        print("="*70)

        figures_dir = scheme['figures_dir']
        figures_dir.mkdir(parents=True, exist_ok=True)
        grid_bins = scheme['grid_bins']
        descriptor_names = ['num_atoms', 'num_bonds']  # Use descriptors for binning

        all_algo_data = {}

        for algo_name, data_by_seed in all_data_by_algo.items():
            print(f"\n{algo_name.upper()}:")

            max_gen = max(max(m['generation'] for m in mols) for mols in data_by_seed.values())
            print(f"  Max generation: {max_gen}")
            print(f"  Computing QD metrics with {scheme['description']}...")

            generations, metrics_by_seed, grid_bounds = compute_qd_metrics(
                data_by_seed, max_gen, is_multiobjective=True,
                ref_point=ref_point, maximize_mask=maximize_mask,
                grid_bins=grid_bins, eval_interval=1,
                descriptor_names=descriptor_names,
                global_ref_for_hv=shared_global_ref  # Use shared reference point
            )

            all_algo_data[algo_name] = (generations, metrics_by_seed)

            print(f"  Generating plots...")
            algo_output_dir = figures_dir / "individual_algorithms" / algo_name
            algo_output_dir.mkdir(parents=True, exist_ok=True)

            # Generate plots with generations on x-axis
            plot_qd_metrics(generations, metrics_by_seed, algo_name,
                           is_multiobjective=True, output_dir=algo_output_dir,
                           x_axis='generations')
            # Generate plots with evaluations on x-axis
            plot_qd_metrics(generations, metrics_by_seed, algo_name,
                           is_multiobjective=True, output_dir=algo_output_dir,
                           x_axis='evaluations')

            # Generate archive heatmaps (individual seeds) using global grid bounds
            generate_archive_heatmaps(
                data_by_seed=data_by_seed,
                algorithm_name=algo_name,
                ref_point=ref_point,
                maximize_mask=maximize_mask,
                grid_bins=grid_bins,
                grid_bounds=grid_bounds,
                output_dir=algo_output_dir,
                descriptor_names=descriptor_names
            )

            # Generate median archive heatmap (across all seeds) using global grid bounds
            generate_median_archive_heatmap(
                data_by_seed=data_by_seed,
                algorithm_name=algo_name,
                ref_point=ref_point,
                maximize_mask=maximize_mask,
                grid_bins=grid_bins,
                grid_bounds=grid_bounds,
                output_dir=algo_output_dir,
                descriptor_names=descriptor_names
            )

        # === COMPARISON PLOTS ===
        print("\n" + "="*70)
        print(f"GENERATING COMPARISON PLOTS - {scheme['name']} BINNING")
        print("="*70)

        if all_algo_data:
            # Create comparison output directory
            comparison_dir = figures_dir / "comparisons"
            comparison_dir.mkdir(parents=True, exist_ok=True)

            # Generate comparison plots for all metrics
            all_metrics = [
                'occupied_cells',
                'beta_gamma_ratio',
                'total_energy_atom_ratio',
                'alpha_range_distance',
                'homo_lumo_gap_range_distance',
                'global_hv',
                'moqd_score'
            ]

            print("\n  Generating standard comparison plots (Median + IQR)...")
            for metric_name in all_metrics:
                # Generations x-axis
                output_file = comparison_dir / f"algorithm_comparison_{metric_name}.png"
                plot_comparison(all_algo_data, metric_name, is_multiobjective=True,
                              output_file=output_file, x_axis='generations')
                # Evaluations x-axis
                output_file = comparison_dir / f"algorithm_comparison_{metric_name}_evals.png"
                plot_comparison(all_algo_data, metric_name, is_multiobjective=True,
                              output_file=output_file, x_axis='evaluations')

            print("\n  Generating mean+median comparison plots (Mean + Median, no IQR)...")
            for metric_name in all_metrics:
                # Generations x-axis
                output_file = comparison_dir / f"algorithm_comparison_{metric_name}_mean_median.png"
                plot_comparison_mean_median(all_algo_data, metric_name, is_multiobjective=True,
                              output_file=output_file, x_axis='generations')
                # Evaluations x-axis
                output_file = comparison_dir / f"algorithm_comparison_{metric_name}_mean_median_evals.png"
                plot_comparison_mean_median(all_algo_data, metric_name, is_multiobjective=True,
                              output_file=output_file, x_axis='evaluations')

    # === FINAL SUMMARY ===
    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)

    print(f"\nFigures saved to TWO separate directories for fair comparison:")
    print(f"\n  📁 {figures_base_dir}/figures_coarse/ (10x10 binning)")
    print(f"    ├── individual_algorithms/")
    print(f"    │   ├── map_elites-fine/")
    print(f"    │   ├── map_elites-coarse/")
    print(f"    │   ├── mome-fine/")
    print(f"    │   ├── mome-coarse/")
    print(f"    │   ├── nsga2/")
    print(f"    │   ├── sa/")
    print(f"    │   └── mu_lambda/")
    print(f"    └── comparisons/")
    print(f"\n  📁 {figures_base_dir}/figures_fine/ (20x20 binning)")
    print(f"    ├── individual_algorithms/")
    print(f"    │   └── (same structure as above)")
    print(f"    └── comparisons/")
    print(f"\nAll algorithms analyzed with IDENTICAL binning schemes for fair comparison!")


if __name__ == "__main__":
    main()