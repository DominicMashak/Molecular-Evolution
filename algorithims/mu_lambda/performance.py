"""
Performance tracking and plotting for (μ+λ) Evolution Strategy
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List


class PerformancePlotter:
    """Create plots for (μ+λ) optimization performance"""

    def __init__(self, output_dir: str):
        """
        Initialize plotter

        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def plot_convergence(self, metrics_file: str = None, metrics_dict: Dict = None):
        """
        Plot fitness convergence over generations

        Args:
            metrics_file: Path to metrics JSON file (if metrics_dict not provided)
            metrics_dict: Dictionary with metrics (if metrics_file not provided)
        """
        # Load metrics
        if metrics_dict is None:
            if metrics_file is None:
                metrics_file = self.output_dir / "performance_metrics.json"
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
        else:
            metrics = metrics_dict

        generations = metrics['generations']
        best_fitness = metrics['best_fitness']
        mean_fitness = metrics['mean_fitness']
        std_fitness = metrics['std_fitness']

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot best and mean fitness
        ax.plot(generations, best_fitness, 'b-', linewidth=2, label='Best Fitness')
        ax.plot(generations, mean_fitness, 'g--', linewidth=2, label='Mean Fitness')

        # Plot standard deviation as shaded region
        mean_arr = np.array(mean_fitness)
        std_arr = np.array(std_fitness)
        ax.fill_between(generations,
                        mean_arr - std_arr,
                        mean_arr + std_arr,
                        alpha=0.2, color='green', label='±1 Std Dev')

        ax.set_xlabel('Generation', fontsize=12)
        ax.set_ylabel('Fitness', fontsize=12)
        ax.set_title('(μ+λ) Evolution Strategy - Fitness Convergence', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save
        plot_file = self.output_dir / "convergence.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved convergence plot to {plot_file}")

    def plot_fitness_distribution(self, population_file: str):
        """
        Plot fitness distribution of final population

        Args:
            population_file: Path to population JSON file
        """
        with open(population_file, 'r') as f:
            pop_data = json.load(f)

        individuals = pop_data['individuals']
        fitnesses = [ind['fitness'] for ind in individuals]

        # Create histogram
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.hist(fitnesses, bins=20, edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(fitnesses), color='r', linestyle='--',
                   linewidth=2, label=f'Mean: {np.mean(fitnesses):.4e}')
        ax.axvline(np.median(fitnesses), color='g', linestyle='--',
                   linewidth=2, label=f'Median: {np.median(fitnesses):.4e}')

        ax.set_xlabel('Fitness', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'Fitness Distribution - Generation {pop_data["generation"]}',
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save
        plot_file = self.output_dir / f"fitness_distribution_gen_{pop_data['generation']:04d}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved fitness distribution to {plot_file}")

    def plot_diversity(self, metrics_file: str = None):
        """
        Plot population diversity (std dev) over generations

        Args:
            metrics_file: Path to metrics JSON file
        """
        if metrics_file is None:
            metrics_file = self.output_dir / "performance_metrics.json"

        with open(metrics_file, 'r') as f:
            metrics = json.load(f)

        generations = metrics['generations']
        std_fitness = metrics['std_fitness']

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(generations, std_fitness, 'purple', linewidth=2)
        ax.set_xlabel('Generation', fontsize=12)
        ax.set_ylabel('Standard Deviation (Fitness)', fontsize=12)
        ax.set_title('Population Diversity Over Time', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save
        plot_file = self.output_dir / "diversity.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved diversity plot to {plot_file}")

    def create_all_plots(self):
        """Create all standard plots"""
        # Convergence plot
        metrics_file = self.output_dir / "performance_metrics.json"
        if metrics_file.exists():
            self.plot_convergence()
            self.plot_diversity()

        # Find latest population file
        pop_files = sorted(self.output_dir.glob("population_gen_*.json"))
        if pop_files:
            self.plot_fitness_distribution(str(pop_files[-1]))


def plot_comparison(results_dirs: List[str], labels: List[str] = None,
                   output_file: str = "comparison.png"):
    """
    Compare multiple (μ+λ) runs

    Args:
        results_dirs: List of result directories to compare
        labels: Labels for each run (if None, use directory names)
        output_file: Output file path
    """
    if labels is None:
        labels = [Path(d).name for d in results_dirs]

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    colors = plt.cm.tab10(np.linspace(0, 1, len(results_dirs)))

    for i, (results_dir, label) in enumerate(zip(results_dirs, labels)):
        metrics_file = Path(results_dir) / "performance_metrics.json"

        if not metrics_file.exists():
            print(f"Warning: {metrics_file} not found, skipping...")
            continue

        with open(metrics_file, 'r') as f:
            metrics = json.load(f)

        generations = metrics['generations']
        best_fitness = metrics['best_fitness']
        mean_fitness = metrics['mean_fitness']

        # Plot best fitness
        axes[0].plot(generations, best_fitness, color=colors[i],
                    linewidth=2, label=label)

        # Plot mean fitness
        axes[1].plot(generations, mean_fitness, color=colors[i],
                    linewidth=2, label=label)

    axes[0].set_xlabel('Generation', fontsize=12)
    axes[0].set_ylabel('Best Fitness', fontsize=12)
    axes[0].set_title('Best Fitness Comparison', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Generation', fontsize=12)
    axes[1].set_ylabel('Mean Fitness', fontsize=12)
    axes[1].set_title('Mean Fitness Comparison', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved comparison plot to {output_file}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python performance.py <results_dir>")
        sys.exit(1)

    results_dir = sys.argv[1]
    plotter = PerformancePlotter(results_dir)
    plotter.create_all_plots()
