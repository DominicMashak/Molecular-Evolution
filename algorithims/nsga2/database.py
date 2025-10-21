"""
Database management for multi-objective NSGA-II
Handles storage of molecules with multiple objectives
"""

def update_molecule_database(self, individuals):
    """
    Update the molecule database with new individuals, supporting multiple objectives.
    
    For each individual, checks if a molecule with the same SMILES already exists.
    If it does and the individual's generation is lower, updates the existing entry.
    If not, appends a new entry with all objectives and properties.
    
    Args:
        individuals (list): List of individual objects with attributes:
            - smiles: SMILES string
            - objectives: List of objective values
            - generation: Generation number
            - rank: Pareto rank (optional)
            - Additional properties (homo_lumo_gap, gamma, etc.)
    """
    for ind in individuals:
        existing = next((m for m in self.all_molecules if m['smiles'] == ind.smiles), None)
        
        if existing:
            # Update if this is from an earlier generation
            if ind.generation < existing['generation']:
                existing['generation'] = ind.generation
                existing['objectives'] = ind.objectives
                existing['rank'] = ind.rank if ind.rank != float('inf') else 999999
                
                # Update named objectives for clarity
                if hasattr(self, 'objectives'):
                    for i, obj_name in enumerate(self.objectives):
                        if i < len(ind.objectives):
                            existing[obj_name] = ind.objectives[i]
                
                # Update additional properties
                if hasattr(ind, 'homo_lumo_gap'):
                    existing['homo_lumo_gap'] = ind.homo_lumo_gap
                if hasattr(ind, 'gamma'):
                    existing['gamma'] = ind.gamma
                if hasattr(ind, 'transition_dipole'):
                    existing['transition_dipole'] = ind.transition_dipole
                if hasattr(ind, 'oscillator_strength'):
                    existing['oscillator_strength'] = ind.oscillator_strength
        else:
            # Create new entry
            mol_entry = {
                'smiles': ind.smiles,
                'objectives': ind.objectives,
                'generation': ind.generation,
                'rank': ind.rank if ind.rank != float('inf') else 999999
            }
            
            # Add named objectives for clarity
            if hasattr(self, 'objectives'):
                for i, obj_name in enumerate(self.objectives):
                    if i < len(ind.objectives):
                        mol_entry[obj_name] = ind.objectives[i]
            
            # Add legacy fields for backward compatibility
            if hasattr(ind, 'beta_surrogate'):
                mol_entry['beta_surrogate'] = ind.beta_surrogate
            if hasattr(ind, 'natoms'):
                mol_entry['natoms'] = ind.natoms
            
            # Add additional properties
            if hasattr(ind, 'homo_lumo_gap'):
                mol_entry['homo_lumo_gap'] = ind.homo_lumo_gap
            if hasattr(ind, 'gamma'):
                mol_entry['gamma'] = ind.gamma
            if hasattr(ind, 'transition_dipole'):
                mol_entry['transition_dipole'] = ind.transition_dipole
            if hasattr(ind, 'oscillator_strength'):
                mol_entry['oscillator_strength'] = ind.oscillator_strength
            
            self.all_molecules.append(mol_entry)


def save_molecule_database(self):
    """
    Save the complete molecule database to JSON file.
    
    Saves all molecules with their objectives and properties to 
    'all_molecules_database.json' in the output directory.
    """
    db_file = self.output_dir / "all_molecules_database.json"
    import json
    
    # Sort by generation for easier analysis
    sorted_molecules = sorted(self.all_molecules, key=lambda x: x.get('generation', 0))
    
    with open(db_file, 'w') as f:
        json.dump(sorted_molecules, f, indent=2)
    
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Saved {len(sorted_molecules)} molecules to {db_file}")


def export_molecules_csv(self, filename=None):
    """
    Export molecule database to CSV format for easy analysis.
    
    Args:
        filename (str, optional): Output filename. Defaults to 'molecules_export.csv'
    """
    import pandas as pd
    from pathlib import Path
    
    if filename is None:
        filename = 'molecules_export.csv'
    
    # Convert to DataFrame
    df_data = []
    for mol in self.all_molecules:
        row = {
            'smiles': mol['smiles'],
            'generation': mol.get('generation', 0),
            'rank': mol.get('rank', 999999)
        }
        
        # Add objectives (both as list and as individual columns)
        if 'objectives' in mol:
            for i, obj_val in enumerate(mol['objectives']):
                obj_name = self.objectives[i] if hasattr(self, 'objectives') and i < len(self.objectives) else f'obj_{i}'
                row[obj_name] = obj_val
        
        # Add any additional properties
        for key in ['homo_lumo_gap', 'gamma', 'transition_dipole', 'oscillator_strength',
                   'beta_surrogate', 'natoms']:
            if key in mol:
                row[key] = mol[key]
        
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    
    # Save to CSV
    csv_file = self.output_dir / filename
    df.to_csv(csv_file, index=False)
    
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Exported {len(df)} molecules to {csv_file}")
    
    return df


def get_pareto_optimal_molecules(self):
    """
    Extract and return only the Pareto-optimal molecules from the database.
    
    Returns:
        list: List of dictionaries containing Pareto-optimal molecules
    """
    from individual import Individual
    from dominance import fast_non_dominated_sort
    
    # Convert stored molecules to Individual objects
    all_inds = []
    for mol in self.all_molecules:
        if 'objectives' in mol and mol['objectives']:
            ind = Individual(
                smiles=mol['smiles'],
                objectives=mol['objectives'],
                generation=mol.get('generation', 0)
            )
            all_inds.append(ind)
    
    if not all_inds:
        return []
    
    # Perform non-dominated sorting
    fronts = fast_non_dominated_sort(all_inds, self.optimize_objectives)
    
    # Extract Pareto front (rank 0)
    pareto_molecules = []
    if fronts and fronts[0]:
        for ind in fronts[0]:
            # Find corresponding molecule in database
            mol_data = next((m for m in self.all_molecules if m['smiles'] == ind.smiles), None)
            if mol_data:
                pareto_molecules.append(mol_data.copy())
    
    return pareto_molecules


def save_pareto_molecules(self, filename='pareto_molecules.json'):
    """
    Save only the Pareto-optimal molecules to a separate file.
    
    Args:
        filename (str): Output filename
    """
    import json
    
    pareto_mols = self.get_pareto_optimal_molecules()
    
    pareto_file = self.output_dir / filename
    with open(pareto_file, 'w') as f:
        json.dump(pareto_mols, f, indent=2)
    
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Saved {len(pareto_mols)} Pareto-optimal molecules to {pareto_file}")


def get_statistics_summary(self):
    """
    Generate summary statistics for the molecule database.
    
    Returns:
        dict: Dictionary containing summary statistics
    """
    import numpy as np
    
    if not self.all_molecules:
        return {}
    
    stats = {
        'total_molecules': len(self.all_molecules),
        'unique_smiles': len(set(m['smiles'] for m in self.all_molecules)),
        'generations': {
            'min': min(m.get('generation', 0) for m in self.all_molecules),
            'max': max(m.get('generation', 0) for m in self.all_molecules)
        }
    }
    
    # Statistics for each objective
    if hasattr(self, 'objectives'):
        stats['objectives'] = {}
        
        for i, obj_name in enumerate(self.objectives):
            obj_values = []
            for mol in self.all_molecules:
                if 'objectives' in mol and i < len(mol['objectives']):
                    val = mol['objectives'][i]
                    if val is not None:
                        obj_values.append(val)
            
            if obj_values:
                stats['objectives'][obj_name] = {
                    'min': float(np.min(obj_values)),
                    'max': float(np.max(obj_values)),
                    'mean': float(np.mean(obj_values)),
                    'std': float(np.std(obj_values)),
                    'median': float(np.median(obj_values))
                }
    
    # Count molecules in each Pareto rank
    rank_counts = {}
    for mol in self.all_molecules:
        rank = mol.get('rank', 999999)
        rank_counts[rank] = rank_counts.get(rank, 0) + 1
    
    stats['rank_distribution'] = rank_counts
    
    return stats


def save_statistics(self, filename='database_statistics.json'):
    """
    Save database statistics to JSON file.
    
    Args:
        filename (str): Output filename
    """
    import json
    
    stats = self.get_statistics_summary()
    
    stats_file = self.output_dir / filename
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Saved database statistics to {stats_file}")
    
    # Also log key statistics
    logger.info(f"Database contains {stats['total_molecules']} total evaluations")
    logger.info(f"  Unique molecules: {stats['unique_smiles']}")
    logger.info(f"  Generations: {stats['generations']['min']} - {stats['generations']['max']}")
    
    if 'objectives' in stats:
        logger.info("Objective statistics:")
        for obj_name, obj_stats in stats['objectives'].items():
            logger.info(f"  {obj_name}: [{obj_stats['min']:.3e}, {obj_stats['max']:.3e}] "
                       f"(mean={obj_stats['mean']:.3e}, std={obj_stats['std']:.3e})")