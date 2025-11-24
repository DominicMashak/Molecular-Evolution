#!/usr/bin/env python3
"""
Enhanced comparison script with generation-wise metrics tracking.
No recalculation - just loads both existing JSON files.
"""

import json
import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import kendalltau, spearmanr, pearsonr
from typing import List, Dict

# Import PyMoo Hypervolume
try:
    from pymoo.indicators.hv import Hypervolume
    PYMOO_AVAILABLE = True
except ImportError:
    PYMOO_AVAILABLE = False

def calculate_pareto_front_size(objectives):
    """
    Calculate Pareto front size (number of non-dominated solutions).
    Objectives: [beta (max), natoms (min)]
    """
    n = len(objectives)
    if n == 0:
        return 0
    
    is_dominated = [False] * n
    
    for i in range(n):
        if is_dominated[i]:
            continue
        for j in range(n):
            if i == j or is_dominated[j]:
                continue
            # Check if i dominates j
            # For [beta (max), natoms (min)]:
            # i dominates j if: beta_i >= beta_j AND natoms_i <= natoms_j
            # with at least one strict inequality
            if (objectives[i][0] >= objectives[j][0] and objectives[i][1] <= objectives[j][1] and
                (objectives[i][0] > objectives[j][0] or objectives[i][1] < objectives[j][1])):
                is_dominated[j] = True
    
    pareto_front_size = sum(1 for dominated in is_dominated if not dominated)
    return pareto_front_size

def calculate_hypervolume(objectives, reference_point=None):
    """Calculate hypervolume for objectives [beta, natoms]."""
    
    objectives = np.array(objectives)
    
    if reference_point is None:
        reference_point = [0.0, np.max(objectives[:, 1]) + 1.0]
    
    # Transform for PyMoo (minimization framework)
    transformed_ref = np.array([
        -reference_point[0],  # Negate beta for maximization
        reference_point[1]    # Keep natoms as-is for minimization
    ])
    
    # Transform objectives
    transformed_obj = objectives.copy()
    transformed_obj[:, 0] = -transformed_obj[:, 0]  # Negate beta for max
    
    # Calculate HV
    hv_indicator = Hypervolume(ref_point=transformed_ref)
    hv = hv_indicator(transformed_obj)
    
    return hv

def extract_generation_metrics(data: List[Dict], beta_key='beta'):
    """
    Extract HV and QD metrics per generation.
    Returns dict: {generation: {'hv': hv, 'qd': qd, 'max_beta': max_beta, 'mean_beta': mean_beta}}
    """
    # Group by generation
    generations = {}
    for mol in data:
        gen = mol.get('generation', 0)
        if gen not in generations:
            generations[gen] = []
        
        beta = mol.get(beta_key)
        natoms = mol.get('natoms')
        
        if beta is not None and natoms is not None:
            generations[gen].append({
                'beta': beta,
                'natoms': natoms
            })
    
    # Calculate metrics per generation
    gen_metrics = {}
    for gen in sorted(generations.keys()):
        mols = generations[gen]
        
        if len(mols) == 0:
            continue
        
        # Extract objectives
        objectives = np.array([[m['beta'], m['natoms']] for m in mols])
        
        # Calculate metrics
        hv = calculate_hypervolume(objectives)
        qd = calculate_pareto_front_size(objectives)
        max_beta = np.max(objectives[:, 0])
        mean_beta = np.mean(objectives[:, 0])
        
        gen_metrics[gen] = {
            'hv': hv,
            'qd': qd,
            'max_beta': max_beta,
            'mean_beta': mean_beta,
            'n_molecules': len(mols)
        }
    
    return gen_metrics

def plot_generation_metrics(dft_metrics, mopac_metrics, output_dir='.'):
    """Plot HV, QD, and beta statistics over generations."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Extract data
    dft_gens = sorted(dft_metrics.keys())
    mopac_gens = sorted(mopac_metrics.keys())
    
    dft_hv = [dft_metrics[g]['hv'] for g in dft_gens]
    mopac_hv = [mopac_metrics[g]['hv'] for g in mopac_gens]
    
    dft_qd = [dft_metrics[g]['qd'] for g in dft_gens]
    mopac_qd = [mopac_metrics[g]['qd'] for g in mopac_gens]
    
    dft_max = [dft_metrics[g]['max_beta'] for g in dft_gens]
    mopac_max = [mopac_metrics[g]['max_beta'] for g in mopac_gens]
    
    dft_mean = [dft_metrics[g]['mean_beta'] for g in dft_gens]
    mopac_mean = [mopac_metrics[g]['mean_beta'] for g in mopac_gens]
    
    # Plot 1: Hypervolume over generations
    axes[0, 0].plot(dft_gens, dft_hv, 'o-', label='DFT (HF)', linewidth=2, markersize=6)
    axes[0, 0].plot(mopac_gens, mopac_hv, 's-', label='MOPAC (PM7)', linewidth=2, markersize=6)
    axes[0, 0].set_xlabel('Generation', fontsize=11)
    axes[0, 0].set_ylabel('Hypervolume', fontsize=11)
    axes[0, 0].set_title('Hypervolume over Generations', fontsize=12, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: QD Score over generations
    axes[0, 1].plot(dft_gens, dft_qd, 'o-', label='DFT (HF)', linewidth=2, markersize=6)
    axes[0, 1].plot(mopac_gens, mopac_qd, 's-', label='MOPAC (PM7)', linewidth=2, markersize=6)
    axes[0, 1].set_xlabel('Generation', fontsize=11)
    axes[0, 1].set_ylabel('QD Score (Pareto Front Size)', fontsize=11)
    axes[0, 1].set_title('QD Score over Generations', fontsize=12, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Max Beta over generations
    axes[1, 0].plot(dft_gens, dft_max, 'o-', label='DFT (HF)', linewidth=2, markersize=6)
    axes[1, 0].plot(mopac_gens, mopac_max, 's-', label='MOPAC (PM7)', linewidth=2, markersize=6)
    axes[1, 0].set_xlabel('Generation', fontsize=11)
    axes[1, 0].set_ylabel('Max Beta', fontsize=11)
    axes[1, 0].set_title('Maximum Beta over Generations', fontsize=12, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Mean Beta over generations
    axes[1, 1].plot(dft_gens, dft_mean, 'o-', label='DFT (HF)', linewidth=2, markersize=6)
    axes[1, 1].plot(mopac_gens, mopac_mean, 's-', label='MOPAC (PM7)', linewidth=2, markersize=6)
    axes[1, 1].set_xlabel('Generation', fontsize=11)
    axes[1, 1].set_ylabel('Mean Beta', fontsize=11)
    axes[1, 1].set_title('Mean Beta over Generations', fontsize=12, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = f'{output_dir}/generation_metrics.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def plot_generation_metrics_separate_scales(dft_metrics, mopac_metrics, output_dir='.'):
    """Plot metrics with separate y-axes for DFT and MOPAC (handles outliers better)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    dft_gens = sorted(dft_metrics.keys())
    mopac_gens = sorted(mopac_metrics.keys())
    
    # Plot 1: Hypervolume (separate scales)
    ax1 = axes[0, 0]
    ax2 = ax1.twinx()
    
    dft_hv = [dft_metrics[g]['hv'] for g in dft_gens]
    mopac_hv = [mopac_metrics[g]['hv'] for g in mopac_gens]
    
    line1 = ax1.plot(dft_gens, dft_hv, 'o-', color='#1E90FF', label='DFT (HF)', linewidth=2, markersize=6)
    line2 = ax2.plot(mopac_gens, mopac_hv, 's-', color='#DC143C', label='MOPAC (PM7)', linewidth=2, markersize=6)
    
    ax1.set_xlabel('Generation', fontsize=11)
    ax1.set_ylabel('DFT Hypervolume', fontsize=11, color='#1E90FF')
    ax2.set_ylabel('MOPAC Hypervolume', fontsize=11, color='#DC143C')
    ax1.tick_params(axis='y', labelcolor='#1E90FF')
    ax2.tick_params(axis='y', labelcolor='#DC143C')
    ax1.set_title('Hypervolume over Generations (Separate Scales)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, fontsize=10, loc='upper left')
    
    # Plot 2: QD Score (same scale is fine)
    axes[0, 1].plot(dft_gens, [dft_metrics[g]['qd'] for g in dft_gens], 'o-', 
                    label='DFT (HF)', linewidth=2, markersize=6)
    axes[0, 1].plot(mopac_gens, [mopac_metrics[g]['qd'] for g in mopac_gens], 's-', 
                    label='MOPAC (PM7)', linewidth=2, markersize=6)
    axes[0, 1].set_xlabel('Generation', fontsize=11)
    axes[0, 1].set_ylabel('QD Score', fontsize=11)
    axes[0, 1].set_title('QD Score over Generations', fontsize=12, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Max Beta (separate scales)
    ax3 = axes[1, 0]
    ax4 = ax3.twinx()
    
    dft_max = [dft_metrics[g]['max_beta'] for g in dft_gens]
    mopac_max = [mopac_metrics[g]['max_beta'] for g in mopac_gens]
    
    line3 = ax3.plot(dft_gens, dft_max, 'o-', color='#1E90FF', label='DFT (HF)', linewidth=2, markersize=6)
    line4 = ax4.plot(mopac_gens, mopac_max, 's-', color='#DC143C', label='MOPAC (PM7)', linewidth=2, markersize=6)
    
    ax3.set_xlabel('Generation', fontsize=11)
    ax3.set_ylabel('DFT Max Beta', fontsize=11, color='#1E90FF')
    ax4.set_ylabel('MOPAC Max Beta', fontsize=11, color='#DC143C')
    ax3.tick_params(axis='y', labelcolor='#1E90FF')
    ax4.tick_params(axis='y', labelcolor='#DC143C')
    ax3.set_title('Maximum Beta over Generations (Separate Scales)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    lines = line3 + line4
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, fontsize=10, loc='upper left')
    
    # Plot 4: Mean Beta (separate scales)
    ax5 = axes[1, 1]
    ax6 = ax5.twinx()
    
    dft_mean = [dft_metrics[g]['mean_beta'] for g in dft_gens]
    mopac_mean = [mopac_metrics[g]['mean_beta'] for g in mopac_gens]
    
    line5 = ax5.plot(dft_gens, dft_mean, 'o-', color='#1E90FF', label='DFT (HF)', linewidth=2, markersize=6)
    line6 = ax6.plot(mopac_gens, mopac_mean, 's-', color='#DC143C', label='MOPAC (PM7)', linewidth=2, markersize=6)
    
    ax5.set_xlabel('Generation', fontsize=11)
    ax5.set_ylabel('DFT Mean Beta', fontsize=11, color='#1E90FF')
    ax6.set_ylabel('MOPAC Mean Beta', fontsize=11, color='#DC143C')
    ax5.tick_params(axis='y', labelcolor='#1E90FF')
    ax6.tick_params(axis='y', labelcolor='#DC143C')
    ax5.set_title('Mean Beta over Generations (Separate Scales)', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    lines = line5 + line6
    labels = [l.get_label() for l in lines]
    ax5.legend(lines, labels, fontsize=10, loc='upper left')
    
    plt.tight_layout()
    output_path = f'{output_dir}/generation_metrics_separate_scales.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def print_generation_summary(dft_metrics, mopac_metrics):
    """Print a summary table of metrics by generation."""
    print(f"\n{'='*100}")
    print(f"Generation-wise Metrics Summary")
    print(f"{'='*100}")
    print(f"{'Gen':>4} | {'DFT HV':>15} | {'MOPAC HV':>15} | {'DFT QD':>7} | {'MOPAC QD':>9} | {'DFT Max β':>12} | {'MOPAC Max β':>12}")
    print(f"{'-'*100}")
    
    all_gens = sorted(set(dft_metrics.keys()) | set(mopac_metrics.keys()))
    
    for gen in all_gens:
        dft = dft_metrics.get(gen, {})
        mopac = mopac_metrics.get(gen, {})
        
        dft_hv = dft.get('hv', 0)
        mopac_hv = mopac.get('hv', 0)
        dft_qd = dft.get('qd', 0)
        mopac_qd = mopac.get('qd', 0)
        dft_max = dft.get('max_beta', 0)
        mopac_max = mopac.get('max_beta', 0)
        
        print(f"{gen:4d} | {dft_hv:15.2f} | {mopac_hv:15.2e} | {dft_qd:7d} | {mopac_qd:9d} | {dft_max:12.2f} | {mopac_max:12.2e}")
    
    print(f"{'='*100}\n")

def main():
    if len(sys.argv) < 3:
        print("Usage: python compare_generations.py <dft_json> <mopac_json> [output_dir]")
        print("\nExample:")
        print("  python compare_generations.py all_molecules_database.json all_molecules_database_new.json ./output")
        sys.exit(1)
    
    dft_json = sys.argv[1]
    mopac_json = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else './output'
    
    if not PYMOO_AVAILABLE:
        print("Warning: PyMoo not available. Installing...")
        os.system("pip install pymoo --break-system-packages -q")
        print("PyMoo installed. Please re-run the script.")
        sys.exit(0)
    
    # Load data
    print(f"Loading DFT data from {dft_json}...")
    with open(dft_json, 'r') as f:
        dft_data = json.load(f)
    print(f"  Loaded {len(dft_data)} molecules")
    
    print(f"\nLoading MOPAC data from {mopac_json}...")
    with open(mopac_json, 'r') as f:
        mopac_data = json.load(f)
    print(f"  Loaded {len(mopac_data)} molecules")
    
    # Extract generation metrics
    print("\nCalculating generation-wise metrics...")
    dft_metrics = extract_generation_metrics(dft_data, beta_key='beta')
    mopac_metrics = extract_generation_metrics(mopac_data, beta_key='beta')
    
    print(f"  DFT: {len(dft_metrics)} generations")
    print(f"  MOPAC: {len(mopac_metrics)} generations")
    
    # Print summary table
    print_generation_summary(dft_metrics, mopac_metrics)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate plots
    print("Generating plots...")
    plot_generation_metrics(dft_metrics, mopac_metrics, output_dir)
    plot_generation_metrics_separate_scales(dft_metrics, mopac_metrics, output_dir)
    
    # Save metrics to JSON
    output_metrics = {
        'dft': dft_metrics,
        'mopac': mopac_metrics
    }
    
    metrics_path = f'{output_dir}/generation_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(output_metrics, f, indent=2)
    print(f"✓ Saved: {metrics_path}")
    
    print(f"\n{'='*60}")
    print("✓ Generation analysis complete!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()