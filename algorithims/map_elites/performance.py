import json
import numpy as np
from pathlib import Path
from typing import Dict, List
from plotting import HypervolumeCalculator, IGDCalculator
import logging

logger = logging.getLogger(__name__)


class PerformanceTracker:
    """Track performance metrics for MAP-Elites"""
    
    def __init__(self, output_dir: Path, reference_point: List[float] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.metrics = {
            'generation': [],
            'coverage': [],
            'archive_size': [],
            'max_beta': [],
            'mean_beta': [],
            'qd_score': [],  # Average objective value of filled cells
            'hypervolume': [],
            'igd_plus': []
        }
        
        # Initialize calculators
        self.hv_reference = reference_point if reference_point is not None else [0.0, 50, 15.0]
        self.hypervolume_calculator = HypervolumeCalculator(self.hv_reference)
        
        # For IGD+, we need a reference set (Pareto front approximation)
        self.reference_set = []  # Will be set later or use a default
        self.igd_calculator = None
        self.previous_qd_score = 0.0
    
    def update(self, generation: int, archive):
        """Update metrics for current generation"""
        self.metrics['generation'].append(generation)
        self.metrics['coverage'].append(archive.get_coverage())
        self.metrics['archive_size'].append(len(archive))
        
        max_beta = archive.get_max_objective()
        mean_beta = archive.get_mean_objective()
        
        self.metrics['max_beta'].append(max_beta if not np.isinf(max_beta) else 0.0)
        self.metrics['mean_beta'].append(mean_beta if not np.isinf(mean_beta) else 0.0)
        
        # Calculate QD Score (sum of fitness improvements over worst fitness)
        # For beta_mean, worst fitness = 0, so QD score = sum of all beta_mean values
        # This is the total "quality-diversity" - it can only increase or stay same
        all_objectives = archive.objectives[~np.isinf(archive.objectives)]
        qd_score = float(np.sum(all_objectives)) if len(all_objectives) > 0 else 0.0
        
        # Check for monotonicity (QD score should never decrease in MAP-Elites)
        if qd_score < self.previous_qd_score - 1e-6:  # Allow small numerical errors
            logger.warning(f"QD Score decreased from {self.previous_qd_score:.6f} to {qd_score:.6f} at generation {generation}")
        self.previous_qd_score = qd_score
        
        self.metrics['qd_score'].append(qd_score)
        
        # Calculate hypervolume
        hv = self.hypervolume_calculator.calculate(archive)
        self.metrics['hypervolume'].append(hv)
        
        # Calculate IGD+ if reference set is available
        if self.igd_calculator:
            igd = self.igd_calculator.calculate(archive)
            self.metrics['igd_plus'].append(igd)
        else:
            self.metrics['igd_plus'].append(0.0)
    
    def save(self):
        """Save metrics to JSON"""
        metrics_file = self.output_dir / 'performance_metrics.json'
        with open(metrics_file, 'w') as f:
            # Convert numpy types to native Python
            clean_metrics = {}
            for k, v in self.metrics.items():
                if isinstance(v, np.ndarray):
                    clean_metrics[k] = v.tolist()
                else:
                    clean_metrics[k] = v
            json.dump(clean_metrics, f, indent=2)
        
        print(f"Saved performance metrics to {metrics_file}")
    
    def set_reference_set(self, reference_set):
        """Set reference set for IGD+ calculation"""
        self.reference_set = reference_set
        self.igd_calculator = IGDCalculator(reference_set)