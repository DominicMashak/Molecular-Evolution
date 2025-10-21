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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Compatibility classes for optimizer imports
class HypervolumeCalculator:
    """Simple hypervolume calculator compatible with optimizer usage."""
    def __init__(self, reference_point: List[float], optimize_objectives: List[tuple]):
        self.reference_point = reference_point
        self.optimize_objectives = optimize_objectives

    def _transform(self, point: List[float]) -> List[float]:
        t = []
        for i, (opt_type, target) in enumerate(self.optimize_objectives):
            val = point[i]
            if opt_type == 'max':
                t.append(val)
            elif opt_type == 'min':
                t.append(-val)
            elif opt_type == 'target':
                t.append(-abs(val - (target if target is not None else 0.0)))
            else:
                t.append(val)
        return t

    def calculate(self, population) -> float:
        """Calculate hypervolume for populations of Individuals with .objectives"""
        if not population:
            return 0.0
        points = [getattr(ind, 'objectives', None) for ind in population]
        points = [p for p in points if p]
        if not points:
            return 0.0
        n_obj = len(points[0])
        # simple 2D and 3D approximation
        transformed = [self._transform(p) for p in points]
        ref = []
        for i, (opt_type, target) in enumerate(self.optimize_objectives[:n_obj]):
            rp = self.reference_point[i] if i < len(self.reference_point) else 0.0
            if opt_type == 'max':
                ref.append(rp)
            elif opt_type == 'min':
                ref.append(-rp)
            elif opt_type == 'target':
                ref.append(-abs(rp - (target if target is not None else 0.0)))
            else:
                ref.append(rp)

        # Filter points not better than reference
        good = [p for p in transformed if all(p[i] >= ref[i] for i in range(len(ref)))]
        if not good:
            return 0.0

        if n_obj == 2:
            # sort by first objective descending
            good.sort(key=lambda x: -x[0])
            hv = 0.0
            prev_x = ref[0]
            for x, y in good:
                dx = max(0.0, x - prev_x)
                dy = max(0.0, y - ref[1])
                hv += dx * dy
                prev_x = max(prev_x, x)
            return float(hv)
        elif n_obj == 3:
            # approximation: sum of rectangular volumes then correct
            hv = 0.0
            for x, y, z in good:
                dx = max(0.0, x - ref[0])
                dy = max(0.0, y - ref[1])
                dz = max(0.0, z - ref[2])
                hv += dx * dy * dz
            return float(hv * 0.7)  # overlap correction
        else:
            logger.warning(f"Hypervolume not implemented for {n_obj} objectives")
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
            'best_beta_per_atom': []
        }

    def update(self, generation: int, population: List, fronts: List = None, hypervolume: float = None):
        self.metrics['generation'].append(int(generation))
        self.metrics['hypervolume'].append(float(hypervolume) if hypervolume is not None else 0.0)

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
    
    # Load database
    db_file = results_path / 'all_molecules_database.json'
    if db_file.exists():
        with open(db_file, 'r') as f:
            results['database'] = json.load(f)
        logger.info(f"Loaded {len(results['database'])} molecules from database")
    
    # Load Pareto front
    pareto_file = results_path / 'pareto_front_molecules.json'
    if pareto_file.exists():
        with open(pareto_file, 'r') as f:
            results['pareto'] = json.load(f)
        logger.info(f"Loaded {len(results['pareto'])} Pareto-optimal molecules")
    
    # Load performance metrics
    metrics_file = results_path / 'performance_metrics.json'
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            results['metrics'] = json.load(f)
        logger.info("Loaded performance metrics")
    
    # Load statistics
    stats_file = results_path / 'database_statistics.json'
    if stats_file.exists():
        with open(stats_file, 'r') as f:
            results['statistics'] = json.load(f)
        logger.info("Loaded statistics")
    
    return results


def plot_pareto_front_2d(results: Dict, output_dir: Path, obj_x: int = 1, obj_y: int = 0):
    """Plot 2D Pareto front projection"""
    if 'pareto' not in results or not results['pareto']:
        logger.warning("No Pareto front data available")
        return
    
    pareto = results['pareto']
    database = results.get('database', [])
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot all molecules
    if database:
        all_x = [m['objectives'][obj_x] for m in database if 'objectives' in m and len(m['objectives']) > max(obj_x, obj_y)]
        all_y = [m['objectives'][obj_y] for m in database if 'objectives' in m and len(m['objectives']) > max(obj_x, obj_y)]
        ax.scatter(all_x, all_y, c='lightblue', s=30, alpha=0.5, label='All molecules')
    
    # Plot Pareto front
    pareto_x = [m['objectives'][obj_x] for m in pareto if 'objectives' in m and len(m['objectives']) > max(obj_x, obj_y)]
    pareto_y = [m['objectives'][obj_y] for m in pareto if 'objectives' in m and len(m['objectives']) > max(obj_x, obj_y)]
    ax.scatter(pareto_x, pareto_y, c='red', s=100, alpha=0.8, edgecolors='k', linewidths=1, label='Pareto front')
    
    # Get objective names if available
    if pareto and 'objectives' in pareto[0]:
        obj_names = [k for k in pareto[0].keys() if k not in ['smiles', 'objectives', 'generation', 'rank']]
        if len(obj_names) > max(obj_x, obj_y):
            xlabel = obj_names[obj_x] if obj_x < len(obj_names) else f'Objective {obj_x+1}'
            ylabel = obj_names[obj_y] if obj_y < len(obj_names) else f'Objective {obj_y+1}'
        else:
            xlabel = f'Objective {obj_x+1}'
            ylabel = f'Objective {obj_y+1}'
    else:
        xlabel = f'Objective {obj_x+1}'
        ylabel = f'Objective {obj_y+1}'
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title('Pareto Front', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = output_dir / 'pareto_front_analysis.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved Pareto front plot to {save_path}")


def plot_convergence(results: Dict, output_dir: Path):
    """Plot convergence curves for all tracked metrics"""
    if 'metrics' not in results:
        logger.warning("No metrics data available")
        return
    
    metrics = results['metrics']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    
    # Hypervolume
    if 'hypervolume' in metrics and metrics['hypervolume']:
        ax = axes[0]
        ax.plot(metrics['generation'], metrics['hypervolume'], 'o-', color='darkblue', linewidth=2)
        ax.fill_between(metrics['generation'], 0, metrics['hypervolume'], alpha=0.3, color='lightblue')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Hypervolume')
        ax.set_title('(a) Hypervolume Indicator')
        ax.grid(True, alpha=0.3)
    
    # Max Beta
    if 'max_beta' in metrics and metrics['max_beta']:
        ax = axes[1]
        ax.plot(metrics['generation'], metrics['max_beta'], 'o-', color='red', linewidth=2, label='Max')
        if 'avg_beta' in metrics:
            ax.plot(metrics['generation'], metrics['avg_beta'], 's--', color='red', linewidth=1.5, alpha=0.7, label='Avg')
        ax.set_xlabel('Generation')
        ax.set_ylabel('β (a.u.)')
        ax.set_title('(b) Hyperpolarizability')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # Atom count
    if 'min_atoms' in metrics and metrics['min_atoms']:
        ax = axes[2]
        ax.plot(metrics['generation'], metrics['min_atoms'], 'o-', color='green', linewidth=2, label='Min')
        if 'avg_atoms' in metrics:
            ax.plot(metrics['generation'], metrics['avg_atoms'], 's--', color='green', linewidth=1.5, alpha=0.7, label='Avg')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Number of Atoms')
        ax.set_title('(c) Molecular Size')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Pareto front size
    if 'pareto_size' in metrics and metrics['pareto_size']:
        ax = axes[3]
        ax.plot(metrics['generation'], metrics['pareto_size'], 'o-', color='purple', linewidth=2)
        ax.fill_between(metrics['generation'], 0, metrics['pareto_size'], alpha=0.3, color='purple')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Pareto Front Size')
        ax.set_title('(d) Non-dominated Solutions')
        ax.grid(True, alpha=0.3)
    
    # Diversity
    if 'population_diversity' in metrics and metrics['population_diversity']:
        ax = axes[4]
        ax.plot(metrics['generation'], metrics['population_diversity'], 'o-', color='orange', linewidth=2)
        ax.set_xlabel('Generation')
        ax.set_ylabel('Diversity')
        ax.set_title('(e) Population Diversity')
        ax.grid(True, alpha=0.3)
    
    # Beta per atom
    if 'best_beta_per_atom' in metrics and metrics['best_beta_per_atom']:
        ax = axes[5]
        ax.plot(metrics['generation'], metrics['best_beta_per_atom'], 'o-', color='brown', linewidth=2)
        ax.set_xlabel('Generation')
        ax.set_ylabel('β/atom (a.u.)')
        ax.set_title('(f) Efficiency Metric')
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    plt.suptitle('Optimization Convergence Analysis', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    save_path = output_dir / 'convergence_analysis.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved convergence analysis to {save_path}")


def plot_objective_distributions(results: Dict, output_dir: Path):
    """Plot distributions of all objectives"""
    if 'database' not in results or not results['database']:
        logger.warning("No database available")
        return
    
    database = results['database']
    pareto = results.get('pareto', [])
    
    # Get number of objectives
    n_objectives = len(database[0]['objectives']) if database and 'objectives' in database[0] else 0
    
    if n_objectives == 0:
        logger.warning("No objectives found in database")
        return
    
    # Get objective names
    obj_names = []
    if database and 'objectives' in database[0]:
        for key in database[0].keys():
            if key not in ['smiles', 'objectives', 'generation', 'rank', 'homo_lumo_gap', 
                          'transition_dipole', 'oscillator_strength', 'gamma']:
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
            report_lines.append(f"\nHypervolume: {initial_hv:.6f} → {final_hv:.6f} ({improvement:+.1f}%)")
        
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