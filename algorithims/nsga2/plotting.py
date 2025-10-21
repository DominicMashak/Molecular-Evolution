"""
Plotting for multi-objective NSGA-II optimization
Supports 2D, 3D, 4D (3D + color), and high-dimensional visualization
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import logging
from matplotlib.ticker import ScalarFormatter
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import numpy as np
from dominance import fast_non_dominated_sort

logger = logging.getLogger(__name__)


def format_objective_name(obj_name):
    """
    Format objective names to be more formal and readable.
    Converts snake_case to Title Case and handles common abbreviations.
    
    Args:
        obj_name: Raw objective name string
        
    Returns:
        Formatted objective name
    """
    replacements = {
        'beta': 'Beta',
        'natoms': 'Number of Atoms',
        'n_atoms': 'Number of Atoms',
        'num_atoms': 'Number of Atoms',
        'qed': 'QED',
        'sa_score': 'SA Score',
        'logp': 'LogP',
        'tpsa': 'TPSA',
        'mw': 'Molecular Weight',
        'molecular_weight': 'Molecular Weight',
        'rssi': 'RSSI',
        'altitude': 'Altitude/Elevation',
        'latitude': 'Latitude',
        'longitude': 'Longitude',
    }
    
    obj_lower = obj_name.lower()
    if obj_lower in replacements:
        return replacements[obj_lower]
    
    return obj_name.replace('_', ' ').title()


def plot_pareto_front(self, population, generation):
    """
    Plot the Pareto front for any number of objectives.
    
    Supports:
    - 2D scatter plot for 2 objectives
    - 3D scatter plot for 3 objectives
    - 4D visualization (3D + color gradient) for 4 objectives
    - Parallel coordinates plot for 5-6 objectives
    - Multiple 3D projections for 7+ objectives
    
    Args:
        population: List of individuals with objectives
        generation: Current generation number
    """
    if not population:
        return

    # Ensure objectives are floats
    for ind in population:
        ind.objectives = [float(x) if x is not None else 0.0 for x in (ind.objectives or [])]

    n_objectives = len(population[0].objectives)
    obj_data = [[float(val) for val in ind.objectives] for ind in population]

    # Compute Pareto front (front 0)
    try:
        fronts = fast_non_dominated_sort(population, getattr(self, 'optimize_objectives', None))
        pareto_front = fronts[0] if fronts else []
        pareto_ids = {id(ind) for ind in pareto_front}
    except Exception as e:
        logger.exception(f"Error computing pareto front for plotting: {e}")
        pareto_ids = set()

    # Helper to format axis labels
    def format_axis_label(idx):
        """Create descriptive axis label with optimization direction"""
        obj_name = self.objectives[idx] if hasattr(self, 'objectives') and idx < len(self.objectives) else f'Obj{idx+1}'
        formatted_name = format_objective_name(obj_name)
        opt_dir = self.optimize_objectives[idx][0] if hasattr(self, 'optimize_objectives') and idx < len(self.optimize_objectives) else 'max'
        return f'{formatted_name} ({opt_dir})'

    # Helper to set scalar formatter
    def set_axis_scalar_format(ax):
        fmt = ScalarFormatter(useOffset=False)
        fmt.set_scientific(False)
        try:
            ax.xaxis.set_major_formatter(fmt)
            ax.yaxis.set_major_formatter(fmt)
        except Exception:
            pass

    if n_objectives == 2:
        # 2D scatter plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        x = np.array([obj[0] for obj in obj_data], dtype=float)
        y = np.array([obj[1] for obj in obj_data], dtype=float)
        
        # Darker blue for non-Pareto points
        non_pareto_mask = [id(ind) not in pareto_ids for ind in population]
        pareto_mask = [id(ind) in pareto_ids for ind in population]
        
        # Non-Pareto points
        ax.scatter(x[non_pareto_mask], y[non_pareto_mask], 
                  c='#1E90FF', s=50, alpha=0.7, edgecolors='k', linewidths=0.5, 
                  label='Non-Pareto Solutions')
        
        # Pareto front points
        if any(pareto_mask):
            ax.scatter(x[pareto_mask], y[pareto_mask], 
                      c='#DC143C', s=120, alpha=0.9, edgecolors='k', linewidths=0.8, 
                      label='Pareto Front')
        
        ax.set_xlabel(format_axis_label(0), fontsize=13, fontweight='bold')
        ax.set_ylabel(format_axis_label(1), fontsize=13, fontweight='bold')
        ax.set_title(f'Pareto Front - Generation {generation}', fontsize=15, fontweight='bold')
        ax.grid(True, alpha=0.3)
        set_axis_scalar_format(ax)
        
        # Invert axes for minimization objectives
        if hasattr(self, 'optimize_objectives'):
            if self.optimize_objectives[0][0] == 'min':
                ax.invert_xaxis()
            if self.optimize_objectives[1][0] == 'min':
                ax.invert_yaxis()
        
        # Legend with box
        legend = ax.legend(loc='best', frameon=True, fancybox=True, shadow=True, 
                          framealpha=0.95, edgecolor='black', facecolor='white')
        legend.get_frame().set_linewidth(1.5)
        
    elif n_objectives == 3:
        # 3D scatter plot
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        x = np.array([obj[0] for obj in obj_data], dtype=float)
        y = np.array([obj[1] for obj in obj_data], dtype=float)
        z = np.array([obj[2] for obj in obj_data], dtype=float)
        
        # color scheme
        non_pareto_mask = [id(ind) not in pareto_ids for ind in population]
        pareto_mask = [id(ind) in pareto_ids for ind in population]
        
        # Non-Pareto points
        ax.scatter(x[non_pareto_mask], y[non_pareto_mask], z[non_pareto_mask],
                  c='#1E90FF', s=40, alpha=0.6, edgecolors='k', linewidths=0.3,
                  label='Non-Pareto Solutions')
        
        # Pareto front points
        if any(pareto_mask):
            ax.scatter(x[pareto_mask], y[pareto_mask], z[pareto_mask],
                      c='#DC143C', s=80, alpha=0.9, edgecolors='k', linewidths=0.5,
                      label='Pareto Front')
        
        ax.set_xlabel(format_axis_label(0), fontsize=12, fontweight='bold')
        ax.set_ylabel(format_axis_label(1), fontsize=12, fontweight='bold')
        ax.set_zlabel(format_axis_label(2), fontsize=12, fontweight='bold')
        ax.set_title(f'Pareto Front - Generation {generation}', fontsize=15, fontweight='bold')
        
        # Format tick labels
        try:
            ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
            ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
            ax.zaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        except Exception:
            pass
        
        # Invert axes for minimization objectives
        if hasattr(self, 'optimize_objectives'):
            if len(self.optimize_objectives) >= 3:
                if self.optimize_objectives[0][0] == 'min':
                    ax.invert_xaxis()
                if self.optimize_objectives[1][0] == 'min':
                    ax.invert_yaxis()
                if self.optimize_objectives[2][0] == 'min':
                    ax.invert_zaxis()
        
        # legend with box
        legend = ax.legend(loc='best', frameon=True, fancybox=True, shadow=True,
                          framealpha=0.95, edgecolor='black', facecolor='white')
        legend.get_frame().set_linewidth(1.5)
        
    elif n_objectives == 4:
        # 4D visualization: 3D plot with color gradient for 4th dimension
        fig = plt.figure(figsize=(14, 11))
        ax = fig.add_subplot(111, projection='3d')
        
        x = np.array([obj[0] for obj in obj_data], dtype=float)
        y = np.array([obj[1] for obj in obj_data], dtype=float)
        z = np.array([obj[2] for obj in obj_data], dtype=float)
        c = np.array([obj[3] for obj in obj_data], dtype=float)
        
        # Normalize color values for the 4th dimension
        c_norm = Normalize(vmin=c.min(), vmax=c.max())
        
        # Choose colormap (viridis, plasma, jet, coolwarm, etc.)
        cmap = plt.cm.viridis
        
        non_pareto_mask = np.array([id(ind) not in pareto_ids for ind in population])
        pareto_mask = np.array([id(ind) in pareto_ids for ind in population])
        
        # Non-Pareto points with color gradient
        if np.any(non_pareto_mask):
            scatter_non_pareto = ax.scatter(
                x[non_pareto_mask], 
                y[non_pareto_mask], 
                z[non_pareto_mask],
                c=c[non_pareto_mask],
                cmap=cmap,
                norm=c_norm,
                s=60,
                alpha=0.6,
                edgecolors='k',
                linewidths=0.3,
                label='Non-Pareto Solutions'
            )
        
        # Pareto front points with color gradient and larger size
        if np.any(pareto_mask):
            scatter_pareto = ax.scatter(
                x[pareto_mask], 
                y[pareto_mask], 
                z[pareto_mask],
                c=c[pareto_mask],
                cmap=cmap,
                norm=c_norm,
                s=120,
                alpha=0.9,
                edgecolors='k',
                linewidths=0.8,
                marker='o',
                label='Pareto Front'
            )
        
        # Add colorbar for the 4th dimension
        sm = ScalarMappable(cmap=cmap, norm=c_norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, pad=0.1, shrink=0.8)
        cbar.set_label(format_axis_label(3), fontsize=12, fontweight='bold')
        
        ax.set_xlabel(format_axis_label(0), fontsize=12, fontweight='bold')
        ax.set_ylabel(format_axis_label(1), fontsize=12, fontweight='bold')
        ax.set_zlabel(format_axis_label(2), fontsize=12, fontweight='bold')
        ax.set_title(f'4D Pareto Front - Generation {generation}', fontsize=15, fontweight='bold')
        
        # Format tick labels
        try:
            ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
            ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
            ax.zaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        except Exception:
            pass
        
        # Invert axes for minimization objectives
        if hasattr(self, 'optimize_objectives'):
            if len(self.optimize_objectives) >= 3:
                if self.optimize_objectives[0][0] == 'min':
                    ax.invert_xaxis()
                if self.optimize_objectives[1][0] == 'min':
                    ax.invert_yaxis()
                if self.optimize_objectives[2][0] == 'min':
                    ax.invert_zaxis()
        
        # legend with box
        legend = ax.legend(loc='best', frameon=True, fancybox=True, shadow=True,
                          framealpha=0.95, edgecolor='black', facecolor='white')
        legend.get_frame().set_linewidth(1.5)
        
    elif n_objectives <= 6:
        # Parallel coordinates plot for 5-6 objectives
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Normalize data for visualization
        obj_array = np.array(obj_data)
        obj_min = obj_array.min(axis=0)
        obj_max = obj_array.max(axis=0)
        obj_range = obj_max - obj_min
        obj_range[obj_range == 0] = 1.0  # Avoid division by zero
        obj_normalized = (obj_array - obj_min) / obj_range
        
        # Plot lines
        x_positions = np.arange(n_objectives)
        for i, (ind, obj_norm) in enumerate(zip(population, obj_normalized)):
            if id(ind) in pareto_ids:
                ax.plot(x_positions, obj_norm, color='#DC143C', alpha=0.8, linewidth=2, zorder=2)
            else:
                ax.plot(x_positions, obj_norm, color='#1E90FF', alpha=0.5, linewidth=1, zorder=1)
        
        # Set x-axis labels with formatted names
        ax.set_xticks(x_positions)
        formatted_labels = [format_axis_label(i) for i in range(n_objectives)]
        ax.set_xticklabels(formatted_labels, rotation=15, ha='right', fontweight='bold')
        ax.set_ylabel('Normalized Value', fontsize=13, fontweight='bold')
        ax.set_title(f'Parallel Coordinates - Generation {generation}', fontsize=15, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(-0.05, 1.05)
        
        # legend with box
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='#DC143C', linewidth=2.5, label='Pareto Front'),
            Line2D([0], [0], color='#1E90FF', linewidth=1.5, label='Non-Pareto Solutions')
        ]
        legend = ax.legend(handles=legend_elements, loc='upper right', frameon=True, 
                          fancybox=True, shadow=True, framealpha=0.95, 
                          edgecolor='black', facecolor='white')
        legend.get_frame().set_linewidth(1.5)
        
    else:
        # For 7+ objectives: Create multiple 3D projection plots
        n_plots = min(4, n_objectives // 3 + 1)  # Up to 4 subplots
        fig = plt.figure(figsize=(16, 12))
        
        plot_combinations = [
            (0, 1, 2),  # First 3 objectives
        ]
        
        # Add additional combinations
        if n_objectives >= 6:
            plot_combinations.append((3, 4, 5))
        if n_objectives >= 9:
            plot_combinations.append((6, 7, 8))
        if n_objectives >= 4:
            plot_combinations.append((0, 1, 3))  # Alternative view
        
        for plot_idx, (i, j, k) in enumerate(plot_combinations[:n_plots]):
            if k >= n_objectives:
                break
                
            ax = fig.add_subplot(2, 2, plot_idx + 1, projection='3d')
            
            x = np.array([obj[i] for obj in obj_data], dtype=float)
            y = np.array([obj[j] for obj in obj_data], dtype=float)
            z = np.array([obj[k] for obj in obj_data], dtype=float)
            
            non_pareto_mask = [id(ind) not in pareto_ids for ind in population]
            pareto_mask = [id(ind) in pareto_ids for ind in population]
            
            # Non-Pareto points
            ax.scatter(x[non_pareto_mask], y[non_pareto_mask], z[non_pareto_mask],
                      c='#1E90FF', s=25, alpha=0.5, edgecolors='k', linewidths=0.2)
            
            # Pareto front points
            if any(pareto_mask):
                ax.scatter(x[pareto_mask], y[pareto_mask], z[pareto_mask],
                          c='#DC143C', s=50, alpha=0.8, edgecolors='k', linewidths=0.3)
            
            ax.set_xlabel(format_axis_label(i), fontsize=10, fontweight='bold')
            ax.set_ylabel(format_axis_label(j), fontsize=10, fontweight='bold')
            ax.set_zlabel(format_axis_label(k), fontsize=10, fontweight='bold')
            ax.set_title(f'{format_objective_name(self.objectives[i])}, '
                        f'{format_objective_name(self.objectives[j])}, '
                        f'{format_objective_name(self.objectives[k])}', 
                        fontsize=11, fontweight='bold')
            
            # Format tick labels
            try:
                ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
                ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
                ax.zaxis.set_major_formatter(ScalarFormatter(useOffset=False))
            except Exception:
                pass
        
        plt.suptitle(f'Pareto Front Projections - Generation {generation}', 
                    fontsize=15, fontweight='bold', y=0.98)

    plt.tight_layout()
    plot_file = self.output_dir / f'pareto_front_gen_{generation:03d}.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved Pareto front plot to {plot_file}")
    
    # Additionally create a pair plot for detailed analysis (for 4-6 objectives)
    if 4 <= n_objectives <= 6:
        try:
            create_pairplot(population, pareto_ids, self.objectives, self.output_dir, generation)
        except Exception as e:
            logger.warning(f"Could not create pair plot: {e}")


def create_pairplot(population, pareto_ids, objective_names, output_dir, generation):
    """
    Create a seaborn pair plot for detailed multi-objective analysis.
    
    Args:
        population: List of individuals
        pareto_ids: Set of IDs for Pareto front members
        objective_names: List of objective names
        output_dir: Directory to save plot
        generation: Current generation number
    """
    # Prepare data
    data_dict = {}
    n_objectives = len(population[0].objectives)
    
    for i, obj_name in enumerate(objective_names[:n_objectives]):
        formatted_name = format_objective_name(obj_name)
        data_dict[formatted_name] = [ind.objectives[i] for ind in population]
    
    data_dict['Pareto Front'] = [id(ind) in pareto_ids for ind in population]
    
    df = pd.DataFrame(data_dict)
    
    # Create pair plot
    pairplot = sns.pairplot(
        df, 
        hue='Pareto Front',
        palette={True: '#DC143C', False: '#1E90FF'},
        diag_kind='kde',
        plot_kws={'alpha': 0.7, 's': 40, 'edgecolor': 'k', 'linewidth': 0.4},
        diag_kws={'alpha': 0.8}
    )
    
    pairplot.fig.suptitle(f'Objective Space Analysis - Generation {generation}', 
                         fontsize=15, fontweight='bold', y=1.02)
    
    # legend
    for ax in pairplot.axes.flat:
        legend = ax.get_legend()
        if legend:
            legend.set_frame_on(True)
            legend.get_frame().set_facecolor('white')
            legend.get_frame().set_edgecolor('black')
            legend.get_frame().set_linewidth(1.5)
            legend.get_frame().set_alpha(0.95)
    
    # Save
    pair_file = output_dir / f'pairplot_gen_{generation:03d}.png'
    plt.savefig(pair_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved pair plot to {pair_file}")


def plot_objective_evolution(self, output_dir=None):
    """
    Plot the evolution of each objective over generations.
    
    Args:
        output_dir: Directory to save plot (uses self.output_dir if None)
    """
    if output_dir is None:
        output_dir = self.output_dir
    
    if not hasattr(self, 'all_molecules') or not self.all_molecules:
        logger.warning("No molecule data available for evolution plot")
        return
    
    # Extract data
    generations = []
    objective_values = {obj: [] for obj in self.objectives}
    
    for mol in self.all_molecules:
        if 'generation' in mol and 'objectives' in mol:
            generations.append(mol['generation'])
            for i, obj_name in enumerate(self.objectives):
                if i < len(mol['objectives']):
                    objective_values[obj_name].append(mol['objectives'][i])
                else:
                    objective_values[obj_name].append(0.0)
    
    if not generations:
        logger.warning("No generation data available for evolution plot")
        return
    
    # Create plot
    n_objectives = len(self.objectives)
    n_cols = 2
    n_rows = (n_objectives + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4*n_rows))
    if n_objectives == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i, obj_name in enumerate(self.objectives):
        ax = axes[i]
        
        # Group by generation and get statistics
        gen_unique = sorted(set(generations))
        gen_best = []
        gen_avg = []
        
        for gen in gen_unique:
            gen_indices = [j for j, g in enumerate(generations) if g == gen]
            gen_values = [objective_values[obj_name][j] for j in gen_indices]
            
            if self.optimize_objectives[i][0] == 'max':
                gen_best.append(max(gen_values) if gen_values else 0.0)
            else:
                gen_best.append(min(gen_values) if gen_values else 0.0)
            
            gen_avg.append(np.mean(gen_values) if gen_values else 0.0)
        
        # Plot
        ax.plot(gen_unique, gen_best, 'o-', label='Best', color='#DC143C', linewidth=2.5, markersize=6)
        ax.plot(gen_unique, gen_avg, 's--', label='Average', color='#1E90FF', linewidth=2, markersize=5, alpha=0.8)
        
        formatted_name = format_objective_name(obj_name)
        ax.set_xlabel('Generation', fontsize=12, fontweight='bold')
        ax.set_ylabel(formatted_name, fontsize=12, fontweight='bold')
        ax.set_title(f'{formatted_name} Evolution ({self.optimize_objectives[i][0]})', 
                    fontsize=13, fontweight='bold')
        
        # legend
        legend = ax.legend(loc='best', frameon=True, fancybox=True, shadow=True,
                          framealpha=0.95, edgecolor='black', facecolor='white')
        legend.get_frame().set_linewidth(1.5)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_objectives, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    evolution_file = output_dir / 'objective_evolution.png'
    plt.savefig(evolution_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved objective evolution plot to {evolution_file}")


def plot_hypervolume_progress(self, performance_tracker, output_dir=None):
    """
    Plot hypervolume indicator progress over generations.
    
    Args:
        performance_tracker: PerformanceTracker instance
        output_dir: Directory to save plot (uses self.output_dir if None)
    """
    if output_dir is None:
        output_dir = self.output_dir
    
    metrics = performance_tracker.metrics
    
    if not metrics.get('hypervolume'):
        logger.warning("No hypervolume data available")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    generations = metrics['generation']
    hypervolumes = metrics['hypervolume']
    
    ax.plot(generations, hypervolumes, 'o-', color='#1E4D8B', linewidth=2.5, markersize=6)
    ax.fill_between(generations, 0, hypervolumes, alpha=0.3, color='#6495ED')
    
    ax.set_xlabel('Generation', fontsize=13, fontweight='bold')
    ax.set_ylabel('Hypervolume Indicator', fontsize=13, fontweight='bold')
    ax.set_title('Hypervolume Convergence', fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    hv_file = output_dir / 'hypervolume_progress.png'
    plt.savefig(hv_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved hypervolume progress plot to {hv_file}")