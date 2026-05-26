import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import logging
from matplotlib.ticker import ScalarFormatter
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


class PerformancePlotter:
    """Plotting utilities for MAP-Elites performance tracking"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def plot_convergence(self, performance_tracker, filename='convergence.png'):
        """Plot convergence metrics over generations"""
        metrics = performance_tracker.metrics
        
        if not metrics.get('generation'):
            logger.warning("No generation data available for convergence plot")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))  # Increased to 2x3 for QD Score
        axes = axes.flatten()
        
        generations = metrics['generation']
        
        # Coverage
        if 'coverage' in metrics:
            axes[0].plot(generations, metrics['coverage'], 'b-', linewidth=2, label='Coverage')
            axes[0].set_xlabel('Generation')
            axes[0].set_ylabel('Archive Coverage')
            axes[0].set_title('Archive Coverage Over Time')
            axes[0].grid(True, alpha=0.3)
            axes[0].set_ylim(0, 1)
        
        # Max beta_mean
        if 'max_beta' in metrics:
            axes[1].plot(generations, metrics['max_beta'], 'r-', linewidth=2, label='Max Beta')
            axes[1].set_xlabel('Generation')
            axes[1].set_ylabel('Max Beta Mean')
            axes[1].set_title('Maximum Beta Mean Over Time')
            axes[1].grid(True, alpha=0.3)
        
        # Mean beta_mean
        if 'mean_beta' in metrics:
            axes[2].plot(generations, metrics['mean_beta'], 'g-', linewidth=2, label='Mean Beta')
            axes[2].set_xlabel('Generation')
            axes[2].set_ylabel('Mean Beta Mean')
            axes[2].set_title('Mean Beta Mean Over Time')
            axes[2].grid(True, alpha=0.3)
        
        # QD Score
        if 'qd_score' in metrics:
            axes[3].plot(generations, metrics['qd_score'], 'c-', linewidth=2, label='QD Score')
            axes[3].set_xlabel('Generation')
            axes[3].set_ylabel('QD Score (Sum)')
            axes[3].set_title('Quality-Diversity Score Over Time')
            axes[3].grid(True, alpha=0.3)
        
        # Archive size
        if 'archive_size' in metrics:
            axes[4].plot(generations, metrics['archive_size'], 'm-', linewidth=2, label='Archive Size')
            axes[4].set_xlabel('Generation')
            axes[4].set_ylabel('Archive Size')
            axes[4].set_title('Archive Size Over Time')
            axes[4].grid(True, alpha=0.3)
        
        # Hide unused subplot
        axes[5].axis('off')
        
        plt.tight_layout()
        plot_file = self.output_dir / filename
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved convergence plot to {plot_file}")
    
    def plot_hypervolume(self, performance_tracker, filename='hypervolume.png'):
        """Plot hypervolume over generations"""
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
        hv_file = self.output_dir / filename
        plt.savefig(hv_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved hypervolume plot to {hv_file}")
    
    def plot_archive_heatmap(self, archive, filename='archive_heatmap.png'):
        """Plot archive coverage. Uses a scatter plot for CVT archives and a
        grid heatmap for fixed-grid archives."""
        heatmap_file = self.output_dir / filename

        # CVT archive: no natural 2D grid — render as scatter plot
        if hasattr(archive, 'centroids'):
            centroids = archive.centroids  # (n_centroids, n_dims)
            objectives = []
            coords = []
            for i, cell in archive.cells.items():
                if cell is not None:
                    coords.append(centroids[i])
                    objectives.append(cell['objective'])
            if not coords:
                return
            coords = np.array(coords)
            # Use first two dims for 2D scatter (or PCA if >2D)
            if coords.shape[1] >= 2:
                x, y = coords[:, 0], coords[:, 1]
                xlabel = archive.measure_keys[0] if len(archive.measure_keys) > 0 else 'dim_0'
                ylabel = archive.measure_keys[1] if len(archive.measure_keys) > 1 else 'dim_1'
            else:
                x = coords[:, 0]
                y = np.zeros_like(x)
                xlabel = archive.measure_keys[0]
                ylabel = ''
            fig, ax = plt.subplots(figsize=(10, 8))
            sc = ax.scatter(x, y, c=objectives, cmap='viridis', s=60, alpha=0.85)
            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label(archive.objective_key)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(f'CVT Archive ({len(objectives)}/{archive.n_centroids} cells filled)')
            plt.tight_layout()
            plt.savefig(heatmap_file, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved archive scatter plot to {heatmap_file}")
            return

        # Grid archive: standard 2D heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        heatmap = np.full(archive.measure_dims, np.nan)
        for entry in archive.get_all_solutions():
            idx = entry['indices']
            heatmap[idx] = entry['objective']
        im = ax.imshow(heatmap.T, origin='lower', cmap='viridis', aspect='auto')
        ax.set_xlabel(f'{archive.measure_keys[0]} Bin')
        ax.set_ylabel(f'{archive.measure_keys[1]} Bin')
        ax.set_title(f'Archive Heatmap ({archive.objective_key})')
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(archive.objective_key)
        plt.tight_layout()
        plt.savefig(heatmap_file, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved archive heatmap to {heatmap_file}")
    
    def plot_archive_cells_evolution(self, performance_tracker, filename='archive_cells_evolution.png'):
        """Plot number of occupied cells (archive size) over generations"""
        metrics = performance_tracker.metrics
        
        if not metrics.get('archive_size'):
            logger.warning("No archive size data available")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        generations = metrics['generation']
        archive_sizes = metrics['archive_size']
        
        ax.plot(generations, archive_sizes, 's-', color='#8B4513', linewidth=2.5, markersize=6)
        ax.fill_between(generations, 0, archive_sizes, alpha=0.3, color='#D2691E')
        
        ax.set_xlabel('Generation', fontsize=13, fontweight='bold')
        ax.set_ylabel('Number of Occupied Cells', fontsize=13, fontweight='bold')
        ax.set_title('Archive Cells Over Evolution', fontsize=15, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        cells_file = self.output_dir / filename
        plt.savefig(cells_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved archive cells evolution plot to {cells_file}")
    
    def plot_qd_score_evolution(self, performance_tracker, filename='qd_score_evolution.png'):
        """Plot QD Score over generations"""
        metrics = performance_tracker.metrics
        
        if not metrics.get('qd_score'):
            logger.warning("No QD Score data available")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        generations = metrics['generation']
        qd_scores = metrics['qd_score']
        
        ax.plot(generations, qd_scores, 'd-', color='#228B22', linewidth=2.5, markersize=6)
        ax.fill_between(generations, 0, qd_scores, alpha=0.3, color='#32CD32')
        
        ax.set_xlabel('Generation', fontsize=13, fontweight='bold')
        ax.set_ylabel('QD Score (Sum of Beta Mean)', fontsize=13, fontweight='bold')
        ax.set_title('Quality-Diversity Score Over Evolution', fontsize=15, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        qd_file = self.output_dir / filename
        plt.savefig(qd_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved QD Score evolution plot to {qd_file}")
    

class HypervolumeCalculator:
    """Calculate hypervolume for MAP-Elites archive using pymoo"""
    
    def __init__(self, reference_point):
        from pymoo.indicators.hv import HV
        self.reference_point = np.array(reference_point)
        self.hv_calculator = HV(ref_point=-self.reference_point)  # pymoo uses negative for max
    
    def calculate(self, archive):
        """Calculate hypervolume of the archive.

        Supports both grid (MAPElitesArchive) and CVT (CVTMAPElitesArchive)
        archives for any single objective key.
        """
        if len(archive) == 0:
            return 0.0

        # Collect objective values generically across both archive types
        objectives = []
        if hasattr(archive, 'iter_filled_cells'):
            for _, cell in archive.iter_filled_cells():
                obj = cell.get('objective')
                if obj is not None and not np.isinf(obj):
                    objectives.append([float(obj)])
        elif hasattr(archive, 'get_all_solutions'):
            obj_key = getattr(archive, 'objective_key', 'beta_mean')
            for entry in archive.get_all_solutions():
                obj = entry.get('properties', {}).get(obj_key)
                if obj is not None and not np.isinf(obj):
                    objectives.append([float(obj)])

        if not objectives:
            return 0.0

        points = np.array(objectives)  # (N, 1) for single-objective
        # pymoo HV: ref_point was set to -reference_point (for maximisation convention)
        return float(self.hv_calculator.do(-points))

class IGDCalculator:
    """Calculate IGD+ (Inverted Generational Distance Plus)"""
    
    def __init__(self, reference_set):
        self.reference_set = np.array(reference_set)
    
    def calculate(self, archive):
        """Calculate IGD+ for the archive"""
        if len(archive) == 0 or len(self.reference_set) == 0:
            return float('inf')
        
        # Get archive points
        archive_points = []
        for entry in archive.get_all_solutions():
            props = entry['properties']
            point = [
                props.get('beta_mean', 0.0),
                props.get('num_atoms', 0),
                props.get('homo_lumo_gap', 0.0)
            ]
            archive_points.append(point)
        
        archive_points = np.array(archive_points)
        
        total_distance = 0.0
        for ref_point in self.reference_set:
            min_dist = float('inf')
            for arch_point in archive_points:
                dist = np.sum(np.maximum(ref_point - arch_point, 0)**2)**0.5  # IGD+
                min_dist = min(min_dist, dist)
            total_distance += min_dist
        
        return total_distance / len(self.reference_set)