import json
import numpy as np
from pathlib import Path
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class MOMEPerformanceTracker:
    """Track performance metrics for MOME (Multi-Objective MAP-Elites)"""
    
    def __init__(self, output_dir: Path, reference_point: List[float] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.metrics = {
            'generation': [],
            'coverage': [],
            'filled_cells': [],
            'total_solutions': [],
            'moqd_score': [],  # Sum of hypervolumes across all cells
            'global_hypervolume': [],  # Hypervolume of global Pareto front
            'global_front_size': [],  # Size of global Pareto front
            'mean_front_size': [],  # Average size of cell fronts
            'max_objectives': [],  # Max value for each objective
            'mean_objectives': []  # Mean value for each objective
        }
        
        # Reference point for hypervolume
        self.reference_point = np.array(reference_point) if reference_point is not None else None
    
    def update(self, generation: int, archive):
        """Update metrics for current generation using PyMoo (same as NSGA-II)"""
        self.metrics['generation'].append(generation)
        self.metrics['coverage'].append(archive.get_coverage())
        self.metrics['filled_cells'].append(archive.n_filled)
        self.metrics['total_solutions'].append(len(archive))

        # MOQD score using PyMoo
        moqd_score = archive.compute_moqd_score()
        self.metrics['moqd_score'].append(float(moqd_score))

        # Global Pareto front metrics using PyMoo
        global_front = archive.get_global_pareto_front()
        self.metrics['global_front_size'].append(len(global_front))

        global_hv = archive.compute_global_hypervolume()
        self.metrics['global_hypervolume'].append(float(global_hv))
        
        # Mean front size per cell
        all_fronts = [archive.fronts[idx] for idx in np.ndindex(archive.measure_dims) 
                     if len(archive.fronts[idx]) > 0]
        mean_size = np.mean([len(f) for f in all_fronts]) if all_fronts else 0.0
        self.metrics['mean_front_size'].append(float(mean_size))
        
        # Objective statistics
        all_solutions = archive.get_all_solutions()
        if all_solutions:
            all_objectives = np.array([s['objectives'] for s in all_solutions])
            max_objs = np.max(all_objectives, axis=0).tolist()
            mean_objs = np.mean(all_objectives, axis=0).tolist()
            self.metrics['max_objectives'].append(max_objs)
            self.metrics['mean_objectives'].append(mean_objs)
        else:
            n_obj = len(archive.objective_keys)
            self.metrics['max_objectives'].append([0.0] * n_obj)
            self.metrics['mean_objectives'].append([0.0] * n_obj)
    
    def save(self):
        """Save metrics to JSON"""
        metrics_file = self.output_dir / 'mome_performance_metrics.json'
        with open(metrics_file, 'w') as f:
            # Convert numpy types to native Python
            clean_metrics = {}
            for k, v in self.metrics.items():
                if isinstance(v, np.ndarray):
                    clean_metrics[k] = v.tolist()
                elif isinstance(v, list) and len(v) > 0:
                    # Handle nested lists/arrays
                    clean_v = []
                    for item in v:
                        if isinstance(item, np.ndarray):
                            clean_v.append(item.tolist())
                        elif isinstance(item, (np.integer, np.floating)):
                            clean_v.append(float(item))
                        else:
                            clean_v.append(item)
                    clean_metrics[k] = clean_v
                else:
                    clean_metrics[k] = v
            json.dump(clean_metrics, f, indent=2)
        
        print(f"Saved MOME performance metrics to {metrics_file}")