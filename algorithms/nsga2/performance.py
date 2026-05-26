#!/usr/bin/env python3
"""
Visualization tool for analyzing NSGA-II optimization results
Generates comprehensive plots and analysis from saved results
"""

import argparse
import json
import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List
from pymoo.indicators.hv import Hypervolume
from matplotlib.ticker import ScalarFormatter
from matplotlib.colors import Normalize
from dominance import fast_non_dominated_sort

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Compatibility classes for optimizer imports
class HypervolumeCalculator:
    """
    Hypervolume calculator compatible with optimizer usage.
    
    PyMoo's Hypervolume class expects:
    1. All objectives to be MINIMIZATION problems
    2. Reference point to be WORSE than all points
    
    This class handles the transformation automatically based on optimize_objectives.
    """
    def __init__(self, reference_point: List[float], optimize_objectives: List[tuple]):
        """
        Initialize hypervolume calculator.
        
        Args:
            reference_point: Reference point for each objective in ORIGINAL space
            optimize_objectives: List of tuples (opt_type, target) where opt_type is 'max', 'min', or 'target'
        """
        self.reference_point = np.array(reference_point, dtype=float)
        self.optimize_objectives = optimize_objectives
        
        # Transform reference point for PyMoo (which expects minimization)
        self.transformed_ref = self._transform_reference_point()
        
        # Create PyMoo hypervolume indicator
        self.hv_indicator = Hypervolume(ref_point=self.transformed_ref)
        
        logger.info(f"HypervolumeCalculator initialized:")
        logger.info(f"  Original reference point: {self.reference_point}")
        logger.info(f"  Transformed reference point: {self.transformed_ref}")
        logger.info(f"  Optimization directions: {[opt[0] for opt in optimize_objectives]}")

    def _transform_reference_point(self):
        """
        Transform reference point for PyMoo's minimization-based hypervolume.
        
        For 'max' objectives: use negative of reference (e.g., 0.0 -> 0.0, but we'll use the negated values)
        For 'min' objectives: use reference as-is
        For 'target' objectives: treat as minimization of distance to target
        """
        transformed = []
        for i, (opt_type, target) in enumerate(self.optimize_objectives):
            ref_val = self.reference_point[i]
            
            if opt_type == 'max':
                # For maximization: negate the reference point
                # The reference should be WORSE (more negative) than all points
                transformed.append(-ref_val)
            elif opt_type == 'min':
                # For minimization: reference point should be LARGER than all points
                transformed.append(ref_val)
            elif opt_type == 'target':
                # For target: we'll treat as minimization of |x - target|
                # Reference is the maximum possible distance
                transformed.append(ref_val)
            else:
                # Default to minimization
                transformed.append(ref_val)
        
        return np.array(transformed, dtype=float)

    def _transform_objectives(self, objectives: np.ndarray) -> np.ndarray:
        """
        Transform objectives for PyMoo's minimization framework.
        
        Args:
            objectives: Array of shape (n_points, n_objectives) in ORIGINAL space
            
        Returns:
            Transformed objectives for minimization
        """
        if len(objectives) == 0:
            return objectives
            
        transformed = objectives.copy()
        
        for i, (opt_type, target) in enumerate(self.optimize_objectives):
            if opt_type == 'max':
                # Negate maximization objectives
                transformed[:, i] = -transformed[:, i]
            elif opt_type == 'target':
                # Convert to distance from target
                transformed[:, i] = np.abs(transformed[:, i] - target)
            # 'min' objectives stay as-is
        
        return transformed

    def calculate(self, population) -> float:
        """
        Calculate hypervolume for population of Individuals with .objectives
        
        Args:
            population: List of Individual objects with .objectives attribute
            
        Returns:
            Hypervolume value (higher is better)
        """
        if not population:
            return 0.0
        
        try:
            # Extract objectives
            objectives = np.array([ind.objectives for ind in population], dtype=float)
            
            if len(objectives) == 0 or objectives.shape[1] == 0:
                return 0.0
            
            # Transform objectives for minimization
            transformed_objectives = self._transform_objectives(objectives)
            
            # Filter out any points that are worse than reference point
            # (PyMoo's HV doesn't include these anyway, but this makes it explicit)
            valid_mask = np.all(transformed_objectives < self.transformed_ref, axis=1)
            
            if not np.any(valid_mask):
                # No valid points (all worse than reference)
                return 0.0
            
            valid_objectives = transformed_objectives[valid_mask]
            
            # Calculate hypervolume
            hv = self.hv_indicator(valid_objectives)
            
            return float(hv)
            
        except Exception as e:
            logger.warning(f"Hypervolume calculation failed: {e}")
            logger.warning(f"  Population size: {len(population)}")
            if len(population) > 0:
                logger.warning(f"  Sample objectives: {population[0].objectives}")
            logger.warning(f"  Reference point: {self.transformed_ref}")
            return 0.0


class PerformanceTracker:
    """Lightweight performance tracker compatible with optimizer usage."""
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.metrics = {
            'generation': [],
            'hypervolume': [],
            'max_beta': [],
            'avg_beta': [],
            'min_atoms': [],
            'avg_atoms': [],
            'pareto_size': [],
            'population_diversity': [],
            'best_beta_per_atom': [],
            'moqd': []  # Add MOQD metric
        }

    def update(self, generation: int, population: List, fronts: List = None, hypervolume: float = None, moqd: float = None):
        self.metrics['generation'].append(int(generation))
        self.metrics['hypervolume'].append(float(hypervolume) if hypervolume is not None else 0.0)
        self.metrics['moqd'].append(float(moqd) if moqd is not None else 0.0)  # Track MOQD

        # Beta values
        betas = [getattr(ind, 'beta_surrogate', None) for ind in population]
        betas = [b for b in betas if b is not None]
        self.metrics['max_beta'].append(max(betas) if betas else 0.0)
        self.metrics['avg_beta'].append(float(np.mean(betas)) if betas else 0.0)

        # Atom counts
        atoms = [getattr(ind, 'natoms', None) for ind in population]
        atoms = [a for a in atoms if a is not None]
        self.metrics['min_atoms'].append(min(atoms) if atoms else 0)
        self.metrics['avg_atoms'].append(float(np.mean(atoms)) if atoms else 0.0)

        # Pareto size
        self.metrics['pareto_size'].append(len(fronts[0]) if fronts else 0)

        # Diversity
        if population and len(population) > 1:
            obj_array = np.array([getattr(ind, 'objectives', []) for ind in population])
            if obj_array.size:
                diversity = float(np.mean(np.std(obj_array, axis=0)))
            else:
                diversity = 0.0
            self.metrics['population_diversity'].append(diversity)
        else:
            self.metrics['population_diversity'].append(0.0)

        # Best beta per atom
        if betas and atoms:
            ratios = [b / a for b, a in zip(betas, atoms) if a > 0]
            self.metrics['best_beta_per_atom'].append(max(ratios) if ratios else 0.0)
        else:
            self.metrics['best_beta_per_atom'].append(0.0)

    def save(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        metrics_file = self.output_dir / 'performance_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        logger.info(f"Saved performance metrics to {metrics_file}")

    def get_dataframe(self):
        return pd.DataFrame(self.metrics)


class PerformancePlotter:
    """Thin wrapper exposing plotting methods expected by optimizer."""
    def __init__(self, output_dir: Path, colors: List[str] = None):
        self.output_dir = Path(output_dir)
        self.colors = colors or ['#2E86AB', '#E63946', '#F77F00', '#06D6A0', '#7209B7']
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.objectives = None
        self.optimize_objectives = None

    def plot_convergence(self, tracker: PerformanceTracker, save_name: str = 'convergence_plot.pdf'):
        """Wrap existing plot_convergence function which expects results dict."""
        results = {'metrics': tracker.metrics}
        try:
            # existing module-level function expects (results, output_dir)
            plot_convergence(results, self.output_dir)
            # also save a PDF copy with requested name if necessary
            default_png = self.output_dir / 'convergence_analysis.png'
            if default_png.exists():
                dest = self.output_dir / save_name
                try:
                    from shutil import copyfile
                    copyfile(default_png, dest.with_suffix('.png'))
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"plot_convergence failed: {e}")

    def plot_hypervolume_comparison(self, trackers: Dict[str, PerformanceTracker], save_name: str = 'hypervolume_comparison.pdf'):
        """Hypervolume comparison plot."""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            for i, (name, tracker) in enumerate(trackers.items()):
                m = tracker.metrics
                gens = m.get('generation', list(range(len(m.get('hypervolume', [])))))
                hv = m.get('hypervolume', [])
                ax.plot(gens, hv, color=self.colors[i % len(self.colors)], linewidth=2, label=name)
            ax.set_xlabel('Generation')
            ax.set_ylabel('Hypervolume')
            ax.set_title('Hypervolume Evolution Comparison')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            save_path = self.output_dir / save_name
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved hypervolume comparison to {save_path}")
        except Exception as e:
            logger.warning(f"plot_hypervolume_comparison failed: {e}")

    def plot_parallel_coordinates(self, population: List, generation: int, save_name: str = None):
        """Create a simple parallel coordinates plot for current population."""
        if not population:
            return
        # Build dataframe
        rows = []
        for ind in population:
            row = {}
            if hasattr(ind, 'beta_surrogate'):
                row['Beta'] = ind.beta_surrogate
            if hasattr(ind, 'natoms'):
                row['Atoms'] = ind.natoms
            if hasattr(ind, 'homo_lumo_gap'):
                row['HOMO-LUMO'] = ind.homo_lumo_gap
            if hasattr(ind, 'objectives'):
                for i, val in enumerate(ind.objectives):
                    row[f'Obj{i+1}'] = val
            rows.append(row)
        df = pd.DataFrame(rows).fillna(0.0)
        # Normalize
        df_norm = (df - df.min()) / (df.max() - df.min() + 1e-12)
        fig, ax = plt.subplots(figsize=(12, 6))
        for _, r in df_norm.iterrows():
            ax.plot(range(len(df_norm.columns)), r.values, color='gray', alpha=0.3)
        ax.set_xticks(range(len(df_norm.columns)))
        ax.set_xticklabels(df_norm.columns, rotation=45)
        ax.set_ylabel('Normalized value')
        ax.set_title(f'Parallel Coordinates - Generation {generation}')
        plt.tight_layout()
        if save_name is None:
            save_name = f'parallel_coordinates_gen_{generation:03d}.png'
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved parallel coordinates plot to {save_path}")

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
        'alpha_mean': 'Alpha',
    }
    
    obj_lower = obj_name.lower()
    if obj_lower in replacements:
        return replacements[obj_lower]
    
    return obj_name.replace('_', ' ').title()


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

# Set publication-quality defaults
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300


def load_results(results_dir: str) -> Dict:
    """Load all result files from optimization directory"""
    results_path = Path(results_dir)
    
    if not results_path.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    
    results = {}
    
    # Load performance metrics
    metrics_file = results_path / 'performance_metrics.json'
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            results['metrics'] = json.load(f)
    
    # Load pareto front
    pareto_file = results_path / 'pareto_front_molecules.json'
    if pareto_file.exists():
        with open(pareto_file, 'r') as f:
            results['pareto'] = json.load(f)
    
    # Load molecule database
    db_file = results_path / 'all_molecules_database.json'
    if db_file.exists():
        with open(db_file, 'r') as f:
            results['database'] = json.load(f)
    
    # Load statistics if available
    stats_file = results_path / 'database_statistics.json'
    if stats_file.exists():
        with open(stats_file, 'r') as f:
            results['statistics'] = json.load(f)
    
    return results


def plot_convergence(results: Dict, output_dir: Path):
    """Plot convergence metrics over generations"""
    if 'metrics' not in results:
        logger.warning("No metrics data found")
        return
    
    metrics = results['metrics']
    
    # Create figure with subplots (expanded to 2x3 for QD bins)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('NSGA-II Convergence Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Hypervolume (now global)
    ax = axes[0, 0]
    if 'hypervolume' in metrics and metrics['hypervolume']:
        generations = metrics.get('generation', list(range(len(metrics['hypervolume']))))
        ax.plot(generations, metrics['hypervolume'], 'b-', linewidth=2, marker='o', markersize=4)
        ax.set_xlabel('Generation')
        ax.set_ylabel('Global Hypervolume')
        ax.set_title('Global Hypervolume Evolution')
        ax.grid(True, alpha=0.3)
    
    # Plot 2: MOQD
    ax = axes[0, 1]
    if 'moqd' in metrics and metrics['moqd']:
        generations = metrics.get('generation', list(range(len(metrics['moqd']))))
        ax.plot(generations, metrics['moqd'], 'g-', linewidth=2, marker='o', markersize=4)
        ax.set_xlabel('Generation')
        ax.set_ylabel('MOQD Score')
        ax.set_title('MOQD Evolution')
        ax.grid(True, alpha=0.3)
    
    # Plot 3: QD Bins
    ax = axes[0, 2]
    if 'qd_bins' in metrics and metrics['qd_bins']:
        generations = metrics.get('generation', list(range(len(metrics['qd_bins']))))
        ax.plot(generations, metrics['qd_bins'], 'purple', linewidth=2, marker='d', markersize=4)
        ax.set_xlabel('Generation')
        ax.set_ylabel('QD Bins')
        ax.set_title('QD Bins Evolution')
        ax.grid(True, alpha=0.3)
    
    # Plot 4: Atoms
    ax = axes[1, 0]
    if 'min_atoms' in metrics and metrics['min_atoms']:
        generations = metrics.get('generation', list(range(len(metrics['min_atoms']))))
        ax.plot(generations, metrics['min_atoms'], 'g-', linewidth=2, label='Min atoms', marker='o', markersize=4)
        if 'avg_atoms' in metrics:
            ax.plot(generations, metrics['avg_atoms'], 'lightgreen', linewidth=2, label='Avg atoms', marker='s', markersize=4)
        ax.set_xlabel('Generation')
        ax.set_ylabel('Number of Atoms')
        ax.set_title('Molecular Size Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 5: Pareto size
    ax = axes[1, 1]
    if 'pareto_size' in metrics and metrics['pareto_size']:
        generations = metrics.get('generation', list(range(len(metrics['pareto_size']))))
        ax.plot(generations, metrics['pareto_size'], 'm-', linewidth=2, label='Pareto size', marker='o', markersize=4)
        ax.set_xlabel('Generation')
        ax.set_ylabel('Pareto Front Size', color='m')
        ax.tick_params(axis='y', labelcolor='m')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
    
    # Plot 6: Population diversity
    ax = axes[1, 2]
    if 'population_diversity' in metrics:
        generations = metrics.get('generation', list(range(len(metrics['population_diversity']))))
        ax.plot(generations, metrics['population_diversity'], 'c-', linewidth=2, label='Diversity', marker='s', markersize=4)
        ax.set_xlabel('Generation')
        ax.set_ylabel('Population Diversity', color='c')
        ax.tick_params(axis='y', labelcolor='c')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = output_dir / 'convergence_analysis.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved convergence analysis to {save_path}")


def plot_pareto_front_2d(results: Dict, output_dir: Path):
    """Plot 2D Pareto front"""
    if 'pareto' not in results or not results['pareto']:
        logger.warning("No Pareto front data found")
        return
    
    pareto = results['pareto']
    
    # Extract objectives - handle both old and new formats
    obj1_vals = []
    obj2_vals = []
    
    for mol in pareto:
        if 'objectives' in mol and len(mol['objectives']) >= 2:
            obj1_vals.append(mol['objectives'][0])
            obj2_vals.append(mol['objectives'][1])
        elif 'beta_surrogate' in mol and 'natoms' in mol:
            # Legacy format
            obj1_vals.append(mol['beta_surrogate'])
            obj2_vals.append(mol['natoms'])
    
    if not obj1_vals:
        logger.warning("No valid objective data in Pareto front")
        return
    
    # Determine objective names from first molecule
    if 'beta' in pareto[0]:
        obj1_name = 'β (a.u.)'
    else:
        obj1_name = 'Objective 1'
    
    if 'natoms' in pareto[0]:
        obj2_name = 'Number of Atoms'
    else:
        obj2_name = 'Objective 2'
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Plot all database molecules if available
    if 'database' in results:
        db = results['database']
        db_obj1 = []
        db_obj2 = []
        for mol in db:
            if 'objectives' in mol and len(mol['objectives']) >= 2:
                db_obj1.append(mol['objectives'][0])
                db_obj2.append(mol['objectives'][1])
            elif 'beta_surrogate' in mol and 'natoms' in mol:
                db_obj1.append(mol['beta_surrogate'])
                db_obj2.append(mol['natoms'])
        
        if db_obj1:
            ax.scatter(db_obj1, db_obj2, c='lightgray', s=50, alpha=0.5, 
                      label='All evaluated molecules', zorder=1)
    
    # Plot Pareto front
    ax.scatter(obj1_vals, obj2_vals, c='red', s=100, alpha=0.8, 
              label='Pareto front', zorder=2, edgecolors='darkred', linewidths=1.5)
    
    # Sort and connect Pareto front
    sorted_indices = sorted(range(len(obj1_vals)), key=lambda i: obj1_vals[i])
    sorted_obj1 = [obj1_vals[i] for i in sorted_indices]
    sorted_obj2 = [obj2_vals[i] for i in sorted_indices]
    ax.plot(sorted_obj1, sorted_obj2, 'r--', alpha=0.5, linewidth=1.5, zorder=2)
    
    ax.set_xlabel(obj1_name, fontsize=12)
    ax.set_ylabel(obj2_name, fontsize=12)
    ax.set_title('Pareto Front - Final Generation', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = output_dir / 'pareto_front_2d.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved Pareto front plot to {save_path}")


def plot_objective_distributions(results: Dict, output_dir: Path):
    """Plot distributions of objectives"""
    if 'database' not in results:
        logger.warning("No database found for distribution plots")
        return
    
    database = results['database']
    pareto = results.get('pareto', [])
    
    # Determine number of objectives
    sample_mol = database[0] if database else None
    if not sample_mol:
        return
    
    if 'objectives' in sample_mol:
        n_objectives = len(sample_mol['objectives'])
    else:
        n_objectives = 2  # Legacy: beta and natoms
    
    # Get objective names
    obj_names = []
    if 'objectives' in sample_mol:
        for key in ['beta', 'natoms', 'homo_lumo_gap', 'energy', 'alpha', 
                          'transition_dipole', 'oscillator_strength', 'gamma']:
            if key in sample_mol:
                obj_names.append(key)
    
    if len(obj_names) < n_objectives:
        obj_names = [f'Objective {i+1}' for i in range(n_objectives)]
    
    # Create subplots
    n_cols = min(3, n_objectives)
    n_rows = (n_objectives + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_objectives == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i in range(n_objectives):
        ax = axes[i]
        
        # Get values
        all_values = [m['objectives'][i] for m in database if 'objectives' in m and len(m['objectives']) > i]
        pareto_values = [m['objectives'][i] for m in pareto if 'objectives' in m and len(m['objectives']) > i]
        
        # Plot distributions
        if all_values:
            ax.hist(all_values, bins=30, alpha=0.5, color='lightblue', label='All molecules', density=True)
        if pareto_values:
            ax.hist(pareto_values, bins=20, alpha=0.7, color='red', label='Pareto front', density=True)
        
        ax.set_xlabel(obj_names[i] if i < len(obj_names) else f'Objective {i+1}')
        ax.set_ylabel('Density')
        ax.set_title(f'{obj_names[i] if i < len(obj_names) else f"Objective {i+1}"} Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    # Hide unused subplots
    for i in range(n_objectives, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    save_path = output_dir / 'objective_distributions.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved objective distributions to {save_path}")


def create_summary_report(results: Dict, output_dir: Path):
    """Create a text summary report"""
    report_lines = []
    
    report_lines.append("="*80)
    report_lines.append("NSGA-II OPTIMIZATION SUMMARY REPORT")
    report_lines.append("="*80)
    report_lines.append("")
    
    # Database statistics
    if 'database' in results:
        db = results['database']
        report_lines.append(f"Total molecules evaluated: {len(db)}")
        unique_smiles = len(set(m['smiles'] for m in db))
        report_lines.append(f"Unique molecules: {unique_smiles}")
        
        generations = [m.get('generation', 0) for m in db]
        report_lines.append(f"Generation range: {min(generations)} - {max(generations)}")
        report_lines.append("")
    
    # Pareto front
    if 'pareto' in results:
        pareto = results['pareto']
        report_lines.append(f"Pareto front size: {len(pareto)}")
        report_lines.append("")
    
    # Objective statistics
    if 'statistics' in results and 'objectives' in results['statistics']:
        report_lines.append("OBJECTIVE STATISTICS:")
        report_lines.append("-"*80)
        
        for obj_name, stats in results['statistics']['objectives'].items():
            report_lines.append(f"\n{obj_name}:")
            report_lines.append(f"  Range: [{stats['min']:.6e}, {stats['max']:.6e}]")
            report_lines.append(f"  Mean ± Std: {stats['mean']:.6e} ± {stats['std']:.6e}")
            report_lines.append(f"  Median: {stats['median']:.6e}")
    
    # Top molecules
    if 'pareto' in results and results['pareto']:
        report_lines.append("\n" + "="*80)
        report_lines.append("TOP 5 PARETO-OPTIMAL MOLECULES:")
        report_lines.append("="*80)
        
        # Sort by first objective (typically beta)
        sorted_pareto = sorted(results['pareto'], 
                              key=lambda x: -x['objectives'][0] if 'objectives' in x else 0)[:5]
        
        for i, mol in enumerate(sorted_pareto, 1):
            report_lines.append(f"\n{i}. {mol['smiles']}")
            if 'objectives' in mol:
                obj_str = ", ".join([f"{val:.6e}" if abs(val) < 0.01 else f"{val:.3f}" 
                                    for val in mol['objectives']])
                report_lines.append(f"   Objectives: [{obj_str}]")
            report_lines.append(f"   Generation: {mol.get('generation', 'N/A')}")
    
    # Performance summary
    if 'metrics' in results:
        metrics = results['metrics']
        report_lines.append("\n" + "="*80)
        report_lines.append("PERFORMANCE SUMMARY:")
        report_lines.append("="*80)
        
        if 'hypervolume' in metrics and metrics['hypervolume']:
            initial_hv = metrics['hypervolume'][0]
            final_hv = metrics['hypervolume'][-1]
            improvement = (final_hv - initial_hv) / (initial_hv + 1e-10) * 100
            report_lines.append(f"\nGlobal Hypervolume: {initial_hv:.6f} → {final_hv:.6f} ({improvement:+.1f}%)")
        
        if 'moqd' in metrics and metrics['moqd']:
            initial_moqd = metrics['moqd'][0]
            final_moqd = metrics['moqd'][-1]
            moqd_improvement = (final_moqd - initial_moqd) / (initial_moqd + 1e-10) * 100
            report_lines.append(f"MOQD Score: {initial_moqd:.6f} → {final_moqd:.6f} ({moqd_improvement:+.1f}%)")
        
        if 'max_beta' in metrics and metrics['max_beta']:
            initial_beta = metrics['max_beta'][0]
            final_beta = metrics['max_beta'][-1]
            improvement = (final_beta - initial_beta) / (initial_beta + 1e-10) * 100
            report_lines.append(f"Max β: {initial_beta:.6e} → {final_beta:.6e} ({improvement:+.1f}%)")
        
        if 'pareto_size' in metrics and metrics['pareto_size']:
            initial_size = metrics['pareto_size'][0]
            final_size = metrics['pareto_size'][-1]
            report_lines.append(f"Pareto size: {initial_size} → {final_size}")
    
    report_lines.append("\n" + "="*80)
    
    # Save report
    report_path = output_dir / 'summary_report.txt'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    # Also print to console
    print('\n'.join(report_lines))
    
    logger.info(f"Saved summary report to {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize and analyze NSGA-II optimization results'
    )
    parser.add_argument('results_dir', type=str,
                       help='Directory containing optimization results')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output directory for visualizations (default: results_dir/analysis)')
    parser.add_argument('--pareto-only', action='store_true',
                       help='Only plot Pareto front')
    parser.add_argument('--no-convergence', action='store_true',
                       help='Skip convergence plots')
    parser.add_argument('--no-distributions', action='store_true',
                       help='Skip distribution plots')
    
    args = parser.parse_args()
    
    # Setup output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path(args.results_dir) / 'analysis'
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    logger.info(f"Analyzing results from: {args.results_dir}")
    logger.info(f"Saving analysis to: {output_dir}")
    
    # Load results
    try:
        results = load_results(args.results_dir)
    except Exception as e:
        logger.error(f"Failed to load results: {e}")
        return
    
    # Generate visualizations
    logger.info("\nGenerating visualizations...")
    
    if not args.pareto_only:
        if not args.no_convergence:
            plot_convergence(results, output_dir)
        
        if not args.no_distributions:
            plot_objective_distributions(results, output_dir)
    
    plot_pareto_front_2d(results, output_dir)
    
    # Create summary report
    create_summary_report(results, output_dir)
    
    logger.info("\n" + "="*80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("="*80)
    logger.info(f"All visualizations saved to: {output_dir}")


if __name__ == '__main__':
    main()