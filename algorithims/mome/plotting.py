import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class MOMEPlotter:
    """Plotting utilities for MOME performance tracking"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def plot_convergence(self, performance_tracker, filename='mome_convergence.png'):
        """Plot convergence metrics over generations"""
        metrics = performance_tracker.metrics
        
        if not metrics.get('generation'):
            logger.warning("No generation data available")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        generations = metrics['generation']
        
        # Coverage
        if 'coverage' in metrics:
            axes[0].plot(generations, metrics['coverage'], 'b-', linewidth=2)
            axes[0].set_xlabel('Generation', fontsize=12)
            axes[0].set_ylabel('Archive Coverage', fontsize=12)
            axes[0].set_title('Archive Coverage Over Time', fontsize=14, fontweight='bold')
            axes[0].grid(True, alpha=0.3)
            axes[0].set_ylim(0, max(metrics['coverage']) * 1.1)
        
        # MOQD Score
        if 'moqd_score' in metrics:
            axes[1].plot(generations, metrics['moqd_score'], 'r-', linewidth=2)
            axes[1].set_xlabel('Generation', fontsize=12)
            axes[1].set_ylabel('MOQD Score', fontsize=12)
            axes[1].set_title('MOQD Score (Sum of Hypervolumes)', fontsize=14, fontweight='bold')
            axes[1].grid(True, alpha=0.3)
            axes[1].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        
        # Global Hypervolume
        if 'global_hypervolume' in metrics:
            axes[2].plot(generations, metrics['global_hypervolume'], 'g-', linewidth=2)
            axes[2].set_xlabel('Generation', fontsize=12)
            axes[2].set_ylabel('Global Hypervolume', fontsize=12)
            axes[2].set_title('Global Pareto Front Hypervolume', fontsize=14, fontweight='bold')
            axes[2].grid(True, alpha=0.3)
            axes[2].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        
        # Total solutions
        if 'total_solutions' in metrics:
            axes[3].plot(generations, metrics['total_solutions'], 'c-', linewidth=2)
            axes[3].set_xlabel('Generation', fontsize=12)
            axes[3].set_ylabel('Total Solutions', fontsize=12)
            axes[3].set_title('Total Solutions in Archive', fontsize=14, fontweight='bold')
            axes[3].grid(True, alpha=0.3)
        
        # Filled cells
        if 'filled_cells' in metrics:
            axes[4].plot(generations, metrics['filled_cells'], 'm-', linewidth=2)
            axes[4].set_xlabel('Generation', fontsize=12)
            axes[4].set_ylabel('Filled Cells', fontsize=12)
            axes[4].set_title('Number of Filled Cells', fontsize=14, fontweight='bold')
            axes[4].grid(True, alpha=0.3)
        
        # Global front size
        if 'global_front_size' in metrics:
            axes[5].plot(generations, metrics['global_front_size'], 'orange', linewidth=2)
            axes[5].set_xlabel('Generation', fontsize=12)
            axes[5].set_ylabel('Global Front Size', fontsize=12)
            axes[5].set_title('Global Pareto Front Size', fontsize=14, fontweight='bold')
            axes[5].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = self.output_dir / filename
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved convergence plot to {plot_file}")
    
    def plot_archive_heatmap_mome(self, archive, reference_point=None, filename='mome_archive_heatmap.png'):
        """Plot heatmap of hypervolumes per cell using PyMoo (same as NSGA-II)"""
        fig, ax = plt.subplots(figsize=(12, 10))

        # Calculate hypervolume for each cell using PyMoo
        heatmap = archive.get_hypervolume_per_cell()
        
        # Replace zeros with nan for better visualization
        heatmap_masked = np.ma.masked_where(heatmap == 0, heatmap)
        
        im = ax.imshow(heatmap_masked.T, origin='lower', cmap='viridis', aspect='auto')
        
        ax.set_xlabel(f'{archive.measure_keys[0]} Bin', fontsize=14)
        ax.set_ylabel(f'{archive.measure_keys[1]} Bin', fontsize=14)
        ax.set_title('MOME Archive Heatmap (Hypervolume per Cell)', fontsize=16, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Hypervolume', fontsize=12)
        
        plt.tight_layout()
        heatmap_file = self.output_dir / filename
        plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved archive heatmap to {heatmap_file}")
    
    def plot_global_pareto_front(self, archive, filename='global_pareto_front.png'):
        """Plot the global Pareto front in objective space"""
        global_front = archive.get_global_pareto_front()
        
        if not global_front or archive.n_objectives != 2:
            logger.warning("Cannot plot global Pareto front")
            return
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Extract objectives
        objectives = np.array([entry['objectives'] for entry in global_front])
        
        # Plot Pareto front
        sorted_indices = np.argsort(objectives[:, 0])
        sorted_objs = objectives[sorted_indices]
        
        ax.plot(sorted_objs[:, 0], sorted_objs[:, 1], 'o-', 
                linewidth=3, markersize=10, label='Global Pareto Front',
                color='#e74c3c', markerfacecolor='#3498db', markeredgewidth=2, markeredgecolor='#2c3e50')
        
        ax.set_xlabel(f'{archive.objective_keys[0]}', fontsize=14, fontweight='bold')
        ax.set_ylabel(f'{archive.objective_keys[1]}', fontsize=14, fontweight='bold')
        ax.set_title('Global Pareto Front', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12)
        
        plt.tight_layout()
        front_file = self.output_dir / filename
        plt.savefig(front_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved global Pareto front to {front_file}")
    
    def plot_pareto_fronts_grid(self, archive, n_samples=9, filename='pareto_fronts_grid.png'):
        """Plot a grid showing Pareto fronts from different cells"""
        # Sample random filled cells
        filled_cells = []
        for idx in np.ndindex(archive.measure_dims):
            if len(archive.fronts[idx]) > 0:
                filled_cells.append(idx)
        
        if not filled_cells:
            logger.warning("No filled cells to plot")
            return
        
        # Sample cells
        n_samples = min(n_samples, len(filled_cells))
        sampled_indices = np.random.choice(len(filled_cells), n_samples, replace=False)
        sampled_cells = [filled_cells[i] for i in sampled_indices]
        
        # Create grid
        rows = int(np.ceil(np.sqrt(n_samples)))
        cols = int(np.ceil(n_samples / rows))
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
        if n_samples == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for i, cell_idx in enumerate(sampled_cells):
            front = archive.fronts[cell_idx]
            objectives = np.array([entry['objectives'] for entry in front])
            
            axes[i].scatter(objectives[:, 0], objectives[:, 1], s=80, alpha=0.7, edgecolors='black', linewidth=1.5)
            axes[i].set_title(f'Cell {cell_idx}', fontsize=11, fontweight='bold')
            axes[i].set_xlabel(archive.objective_keys[0], fontsize=10)
            axes[i].set_ylabel(archive.objective_keys[1], fontsize=10)
            axes[i].grid(True, alpha=0.3)
        
        # Hide extra subplots
        for i in range(n_samples, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle('Pareto Fronts from Different Cells', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        grid_file = self.output_dir / filename
        plt.savefig(grid_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved Pareto fronts grid to {grid_file}")
    
    def plot_objective_histograms(self, archive, filename='objective_histograms.png'):
        """Plot histograms of objective values"""
        all_solutions = archive.get_all_solutions()
        
        if not all_solutions:
            logger.warning("No solutions to plot")
            return
        
        objectives = np.array([s['objectives'] for s in all_solutions])
        
        fig, axes = plt.subplots(1, archive.n_objectives, figsize=(7*archive.n_objectives, 6))
        if archive.n_objectives == 1:
            axes = [axes]
        
        for i, obj_key in enumerate(archive.objective_keys):
            axes[i].hist(objectives[:, i], bins=40, alpha=0.7, edgecolor='black', linewidth=1.2, color='steelblue')
            axes[i].set_xlabel(obj_key, fontsize=13, fontweight='bold')
            axes[i].set_ylabel('Frequency', fontsize=13, fontweight='bold')
            axes[i].set_title(f'Distribution of {obj_key}', fontsize=14, fontweight='bold')
            axes[i].grid(True, alpha=0.3, axis='y')
            
            # Add statistics
            mean_val = np.mean(objectives[:, i])
            std_val = np.std(objectives[:, i])
            axes[i].axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
            axes[i].legend(fontsize=10)
        
        plt.tight_layout()
        hist_file = self.output_dir / filename
        plt.savefig(hist_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved objective histograms to {hist_file}")
    
    def plot_objective_evolution(self, performance_tracker, filename='objective_evolution.png'):
        """Plot how max and mean objectives evolve over generations"""
        metrics = performance_tracker.metrics

        if not metrics.get('generation') or not metrics.get('max_objectives'):
            logger.warning("No objective evolution data available")
            return

        generations = metrics['generation']

        # Handle both dictionary and array formats
        if isinstance(metrics['max_objectives'], dict):
            # Dictionary format: {'obj_0': [...], 'obj_1': [...], ...}
            obj_keys = sorted(metrics['max_objectives'].keys())
            n_objectives = len(obj_keys)
            max_objs = [metrics['max_objectives'][key] for key in obj_keys]
            mean_objs = [metrics['mean_objectives'][key] for key in obj_keys]
        else:
            # Array format
            max_objs = np.array(metrics['max_objectives'])
            mean_objs = np.array(metrics['mean_objectives'])
            n_objectives = max_objs.shape[1]
            obj_keys = [f'Objective {i+1}' for i in range(n_objectives)]
            max_objs = [max_objs[:, i] for i in range(n_objectives)]
            mean_objs = [mean_objs[:, i] for i in range(n_objectives)]

        fig, axes = plt.subplots(1, n_objectives, figsize=(8*n_objectives, 6))
        if n_objectives == 1:
            axes = [axes]

        for i in range(n_objectives):
            axes[i].plot(generations, max_objs[i], 'o-', linewidth=2,
                        markersize=6, label='Max', color='#e74c3c')
            axes[i].plot(generations, mean_objs[i], 's-', linewidth=2,
                        markersize=6, label='Mean', color='#3498db')
            axes[i].set_xlabel('Generation', fontsize=13, fontweight='bold')
            axes[i].set_ylabel(f'Objective {i+1}', fontsize=13, fontweight='bold')
            axes[i].set_title(f'Objective {i+1} Evolution', fontsize=14, fontweight='bold')
            axes[i].grid(True, alpha=0.3)
            axes[i].legend(fontsize=11)

        plt.tight_layout()
        evo_file = self.output_dir / filename
        plt.savefig(evo_file, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved objective evolution plot to {evo_file}")
    
    def plot_all(self, archive, performance_tracker, reference_point=None):
        """Generate all plots at once (reference_point no longer needed - uses PyMoo)"""
        print("\nGenerating comprehensive plots...")

        try:
            self.plot_convergence(performance_tracker)
            print("  ✓ Convergence plot")
        except Exception as e:
            print(f"  ✗ Convergence plot failed: {e}")

        try:
            self.plot_archive_heatmap_mome(archive)
            print("  ✓ Archive heatmap")
        except Exception as e:
            print(f"  ✗ Archive heatmap failed: {e}")
        
        try:
            self.plot_global_pareto_front(archive)
            print("  ✓ Global Pareto front")
        except Exception as e:
            print(f"  ✗ Global Pareto front failed: {e}")
        
        try:
            self.plot_pareto_fronts_grid(archive, n_samples=9)
            print("  ✓ Pareto fronts grid")
        except Exception as e:
            print(f"  ✗ Pareto fronts grid failed: {e}")
        
        try:
            self.plot_objective_histograms(archive)
            print("  ✓ Objective histograms")
        except Exception as e:
            print(f"  ✗ Objective histograms failed: {e}")
        
        try:
            self.plot_objective_evolution(performance_tracker)
            print("  ✓ Objective evolution")
        except Exception as e:
            print(f"  ✗ Objective evolution failed: {e}")
        
        print(f"\nAll plots saved to: {self.output_dir}")


def reconstruct_metrics_from_archives(results_dir: str):
    """
    Reconstruct performance metrics from saved archive JSON files.
    Uses actual PyMoo hypervolume calculations from the archive.

    Args:
        results_dir: Directory containing mome_archive_gen_*.json files

    Returns:
        dict: Reconstructed metrics data
    """
    import sys
    import os
    # Add archive module to path
    sys.path.insert(0, str(Path(__file__).parent))
    import archive as ma

    results_path = Path(results_dir)

    print(f"\nReconstructing metrics from archive files in: {results_dir}")

    # Find all archive files
    archive_files = sorted(results_path.glob('mome_archive_gen_*.json'))

    if not archive_files:
        raise FileNotFoundError(f"No archive files found in {results_dir}")

    print(f"Found {len(archive_files)} archive files")

    # Initialize metrics structure with keys that match plot_convergence expectations
    metrics = {
        'generation': [],
        'total_solutions': [],      # total number of solutions in archive
        'filled_cells': [],         # number of filled cells (same as coverage)
        'coverage': [],             # kept for compatibility
        'global_hypervolume': [],   # hypervolume of global Pareto front
        'moqd_score': [],           # MOQD score (sum of hypervolumes)
        'global_front_size': [],    # size of global Pareto front
        'mean_objectives': {},
        'max_objectives': {}
    }

    # Load first archive to get metadata
    with open(archive_files[0], 'r') as f:
        first_archive = json.load(f)

    # Extract archive configuration
    measure_dims = first_archive.get('measure_dims', [20, 20])
    measure_keys = first_archive.get('measure_keys', ['num_atoms_bin', 'num_bonds_bin'])
    objective_keys = first_archive.get('objective_keys', ['obj_0', 'obj_1', 'obj_2', 'obj_3'])
    optimize_objectives = first_archive.get('optimize_objectives', [['maximize', None]] * len(objective_keys))

    print(f"Archive config: dims={measure_dims}, measures={measure_keys}, objectives={objective_keys}")

    # Process each archive file
    for archive_file in archive_files:
        with open(archive_file, 'r') as f:
            archive_data = json.load(f)

        # Extract generation number from data or filename
        gen_num = archive_data.get('generation', int(archive_file.stem.split('_')[-1]))
        metrics['generation'].append(gen_num)

        # Reconstruct the archive object to use its hypervolume calculations
        archive = ma.MOMEArchive(
            measure_dims=measure_dims,
            measure_keys=measure_keys,
            objective_keys=objective_keys,
            max_front_size=50,
            optimize_objectives=optimize_objectives
        )

        # Populate archive from saved data
        archive_size = 0
        all_objectives = []
        filled_cells = 0

        # Handle different archive structures
        if 'cells' in archive_data:
            # New structure with cells list
            for idx, cell in enumerate(archive_data['cells']):
                if cell and 'solutions' in cell:
                    solutions = cell['solutions']
                    if solutions:
                        filled_cells += 1
                        archive_size += len(solutions)
                        # Reconstruct cell index
                        cell_idx = np.unravel_index(idx, tuple(measure_dims))
                        # Load solutions into archive
                        archive.fronts[cell_idx] = []
                        for solution in solutions:
                            if 'objectives' in solution:
                                objectives_arr = np.array(solution['objectives'])
                                all_objectives.append(solution['objectives'])
                                archive.fronts[cell_idx].append({
                                    'solution': solution.get('solution', ''),
                                    'properties': solution.get('properties', {}),
                                    'objectives': objectives_arr
                                })
        else:
            # Old structure with cell dictionary
            for cell_data in archive_data.values():
                if isinstance(cell_data, list) and len(cell_data) > 0:
                    filled_cells += 1
                    archive_size += len(cell_data)
                    for solution in cell_data:
                        if 'objectives' in solution:
                            all_objectives.append(solution['objectives'])

        # Store metrics with correct key names
        metrics['total_solutions'].append(archive_size)
        metrics['filled_cells'].append(filled_cells)
        metrics['coverage'].append(filled_cells)  # Keep for compatibility

        # Calculate actual global Pareto front size
        global_front = archive.get_global_pareto_front()
        metrics['global_front_size'].append(len(global_front))

        # Calculate objective statistics
        if all_objectives:
            all_objectives_arr = np.array(all_objectives)
            n_objectives = all_objectives_arr.shape[1]

            mean_objs = np.mean(all_objectives_arr, axis=0).tolist()
            max_objs = np.max(all_objectives_arr, axis=0).tolist()

            for i in range(n_objectives):
                obj_key = f'obj_{i}'
                if obj_key not in metrics['mean_objectives']:
                    metrics['mean_objectives'][obj_key] = []
                    metrics['max_objectives'][obj_key] = []
                metrics['mean_objectives'][obj_key].append(mean_objs[i])
                metrics['max_objectives'][obj_key].append(max_objs[i])

        # Calculate ACTUAL hypervolume metrics using PyMoo
        try:
            # Global hypervolume: HV of the global Pareto front
            global_hv = archive.compute_global_hypervolume()
            metrics['global_hypervolume'].append(float(global_hv))

            # MOQD score: sum of hypervolumes across all cells
            moqd = archive.compute_moqd_score()
            metrics['moqd_score'].append(float(moqd))

            print(f"  Gen {gen_num}: Global HV={global_hv:.4f}, MOQD={moqd:.4f}, Solutions={archive_size}")
        except Exception as e:
            print(f"  Warning: Could not calculate hypervolume for gen {gen_num}: {e}")
            metrics['global_hypervolume'].append(0.0)
            metrics['moqd_score'].append(0.0)

    print(f"Reconstructed metrics for {len(metrics['generation'])} generations")

    # Save reconstructed metrics
    metrics_file = results_path / 'mome_performance_metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved reconstructed metrics to: {metrics_file}")

    return metrics


def plot_from_saved_metrics(metrics_file: str, output_dir: str = None):
    """
    Generate plots from saved performance metrics JSON file.
    Useful for standalone plotting after optimization is complete.

    Args:
        metrics_file: Path to mome_performance_metrics.json
        output_dir: Output directory for plots (default: same as metrics file)
    """
    metrics_path = Path(metrics_file)

    # If metrics file doesn't exist, try to reconstruct from archives
    if not metrics_path.exists():
        print(f"Metrics file not found: {metrics_file}")
        results_dir = metrics_path.parent

        # Try to reconstruct from archive files
        try:
            metrics_data = reconstruct_metrics_from_archives(str(results_dir))
            metrics_path = results_dir / 'mome_performance_metrics.json'
        except Exception as e:
            raise FileNotFoundError(f"Could not find or reconstruct metrics file: {e}")
    else:
        # Load existing metrics
        with open(metrics_path, 'r') as f:
            metrics_data = json.load(f)

    if output_dir is None:
        output_dir = metrics_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
    
    # Create a mock performance tracker
    class MockPerformanceTracker:
        def __init__(self, metrics):
            self.metrics = metrics
            self.output_dir = output_dir
    
    tracker = MockPerformanceTracker(metrics_data)
    
    # Create plotter and generate plots
    plotter = MOMEPlotter(output_dir)
    
    print(f"\nGenerating plots from: {metrics_file}")
    
    try:
        plotter.plot_convergence(tracker, filename='mome_convergence_standalone.png')
        print("  ✓ Convergence plot")
    except Exception as e:
        print(f"  ✗ Convergence plot failed: {e}")
    
    try:
        plotter.plot_objective_evolution(tracker, filename='objective_evolution_standalone.png')
        print("  ✓ Objective evolution plot")
    except Exception as e:
        print(f"  ✗ Objective evolution plot failed: {e}")
    
    print(f"\nPlots saved to: {output_dir}")


if __name__ == "__main__":
    import sys
    import os
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python plotting.py <results_directory>")
        print("  python plotting.py <path_to_mome_performance_metrics.json> [output_dir]")
        print("\nExamples:")
        print("  python plotting.py ./mome_results")
        print("  python plotting.py ~/Molecular-Evolution/algorithims/mome/mome_results")
        sys.exit(1)
    
    input_path = sys.argv[1]
    
    # Convert Windows UNC path to proper format if needed
    if '\\wsl.localhost\\' in input_path or '\\\\wsl.localhost\\' in input_path:
        # Extract the Linux path from Windows UNC format
        parts = input_path.replace('\\', '/').split('/')
        try:
            # Find the distro name and reconstruct path
            distro_idx = parts.index('wsl.localhost') + 1
            linux_path = '/' + '/'.join(parts[distro_idx+1:])
            input_path = linux_path
            print(f"Converted Windows UNC path to: {input_path}")
        except (ValueError, IndexError):
            pass
    
    # Check if input is a directory or file
    if os.path.isdir(input_path):
        # It's a directory, look for metrics file
        metrics_file = os.path.join(input_path, 'mome_performance_metrics.json')
        output_dir = input_path
    else:
        # Assume it's a metrics file path
        metrics_file = input_path
        output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    plot_from_saved_metrics(metrics_file, output_dir)