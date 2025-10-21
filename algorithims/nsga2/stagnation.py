"""
Stagnation-based adaptive mutation system for molecular optimization.

This module implements a dynamic mutation strategy that responds to fitness stagnation
by increasing the intensity of atom addition mutations.
"""

import logging
from typing import Dict, List, Tuple
import numpy as np

logger = logging.getLogger(__name__)

class StagnationDetector:
    """Detects and tracks fitness stagnation across generations."""
    
    def __init__(self, 
                 stagnation_threshold: int = 5,
                 improvement_epsilon: float = 1e-6):
        """
        Initialize stagnation detector.
        
        Args:
            stagnation_threshold: Number of generations without improvement to trigger stagnation
            improvement_epsilon: Minimum relative improvement to count as progress
        """
        self.stagnation_threshold = stagnation_threshold
        self.improvement_epsilon = improvement_epsilon
        self.best_fitness_history = []
        self.stagnation_counter = 0
        self.last_improvement_gen = 0
        
    def update(self, generation: int, best_fitness: float) -> Tuple[bool, int]:
        """
        Update with new generation's best fitness.
        
        Args:
            generation: Current generation number
            best_fitness: Best fitness value in current generation
            
        Returns:
            Tuple of (is_stagnant, stagnation_duration)
        """
        self.best_fitness_history.append(best_fitness)
        
        # Check if we have improvement
        if len(self.best_fitness_history) > 1:
            prev_best = max(self.best_fitness_history[:-1])
            current_best = best_fitness
            
            # Calculate relative improvement
            if prev_best > 0:
                relative_improvement = (current_best - prev_best) / abs(prev_best)
            else:
                relative_improvement = abs(current_best - prev_best)
            
            if relative_improvement > self.improvement_epsilon:
                # Significant improvement
                self.stagnation_counter = 0
                self.last_improvement_gen = generation
            else:
                # No significant improvement
                self.stagnation_counter += 1
        
        is_stagnant = self.stagnation_counter >= self.stagnation_threshold
        stagnation_duration = self.stagnation_counter
        
        return is_stagnant, stagnation_duration
    
    def get_statistics(self) -> Dict:
        """Get stagnation statistics."""
        return {
            'stagnation_counter': self.stagnation_counter,
            'last_improvement_gen': self.last_improvement_gen,
            'best_fitness_ever': max(self.best_fitness_history) if self.best_fitness_history else 0.0,
            'generations_tracked': len(self.best_fitness_history)
        }
    
    def reset(self):
        """Reset stagnation counter (e.g., after intervention)."""
        self.stagnation_counter = 0


class StagnationAdaptiveMutation:
    """
    Adaptive mutation system that responds to stagnation by intensifying atom additions.
    
    Implements two strategies:
    1. Increase mutation weight for atom additions
    2. Increase number of atoms added per mutation
    """
    
    def __init__(self,
                 base_add_weight: float = 1.0,
                 base_atoms_per_add: int = 1,
                 max_atoms_per_add: int = 5,
                 stagnation_threshold: int = 5,
                 weight_boost_factor: float = 2.0,
                 atoms_per_stagnation: float = 0.5,
                 use_weight_boost: bool = True,
                 use_atom_boost: bool = True,
                 improvement_epsilon: float = 1e-6):
        """
        Initialize adaptive mutation system.
        
        Args:
            base_add_weight: Base weight for add_atoms mutation
            base_atoms_per_add: Base number of atoms to add per mutation
            max_atoms_per_add: Maximum atoms that can be added in one mutation
            stagnation_threshold: Generations without improvement to trigger adaptation
            weight_boost_factor: Factor to multiply weight by per stagnation period
            atoms_per_stagnation: Additional atoms to add per stagnation period
            use_weight_boost: Enable weight boosting strategy
            use_atom_boost: Enable atom count boosting strategy
            improvement_epsilon: Minimum improvement to reset stagnation
        """
        self.base_add_weight = base_add_weight
        self.base_atoms_per_add = base_atoms_per_add
        self.max_atoms_per_add = max_atoms_per_add
        self.stagnation_threshold = stagnation_threshold
        self.weight_boost_factor = weight_boost_factor
        self.atoms_per_stagnation = atoms_per_stagnation
        self.use_weight_boost = use_weight_boost
        self.use_atom_boost = use_atom_boost
        
        # Initialize stagnation detector
        self.detector = StagnationDetector(
            stagnation_threshold=stagnation_threshold,
            improvement_epsilon=improvement_epsilon
        )
        
        # Current adaptive parameters
        self.current_add_weight = base_add_weight
        self.current_atoms_per_add = base_atoms_per_add
        
        # Statistics
        self.intervention_history = []
        
    def update(self, generation: int, best_fitness: float) -> Dict:
        """
        Update mutation parameters based on current fitness.
        
        Args:
            generation: Current generation number
            best_fitness: Best fitness in current generation
            
        Returns:
            Dictionary with updated mutation parameters and statistics
        """
        is_stagnant, stagnation_duration = self.detector.update(generation, best_fitness)
        
        # Calculate adaptive parameters based on stagnation
        if is_stagnant:
            # Weight boost strategy
            if self.use_weight_boost:
                # Exponential increase with stagnation duration
                weight_multiplier = self.weight_boost_factor ** (stagnation_duration / self.stagnation_threshold)
                self.current_add_weight = self.base_add_weight * weight_multiplier
            else:
                self.current_add_weight = self.base_add_weight
            
            # Atom count boost strategy
            if self.use_atom_boost:
                # Linear increase with stagnation duration
                additional_atoms = int(stagnation_duration * self.atoms_per_stagnation)
                self.current_atoms_per_add = min(
                    self.base_atoms_per_add + additional_atoms,
                    self.max_atoms_per_add
                )
            else:
                self.current_atoms_per_add = self.base_atoms_per_add
            
            # Log intervention
            intervention = {
                'generation': generation,
                'stagnation_duration': stagnation_duration,
                'add_weight': self.current_add_weight,
                'atoms_per_add': self.current_atoms_per_add,
                'best_fitness': best_fitness
            }
            self.intervention_history.append(intervention)
            
            logger.info(f"STAGNATION DETECTED (duration: {stagnation_duration} gens)")
            logger.info(f"   Adaptive response:")
            if self.use_weight_boost:
                logger.info(f"   - Add atom weight: {self.base_add_weight:.2f} → {self.current_add_weight:.2f}")
            if self.use_atom_boost:
                logger.info(f"   - Atoms per addition: {self.base_atoms_per_add} → {self.current_atoms_per_add}")
        else:
            # Reset to base parameters when not stagnant
            self.current_add_weight = self.base_add_weight
            self.current_atoms_per_add = self.base_atoms_per_add
        
        return {
            'is_stagnant': is_stagnant,
            'stagnation_duration': stagnation_duration,
            'add_weight': self.current_add_weight,
            'atoms_per_add': self.current_atoms_per_add,
            'use_weight_boost': self.use_weight_boost,
            'use_atom_boost': self.use_atom_boost
        }
    
    def get_mutation_weights(self, base_weights: Dict[str, float]) -> Dict[str, float]:
        """
        Get adjusted mutation weights based on current stagnation state.
        
        Args:
            base_weights: Original mutation weights dictionary
            
        Returns:
            Adjusted mutation weights
        """
        adjusted_weights = base_weights.copy()
        
        # Apply weight boost to add_atom mutations (types 2 and 3: add_atom_inline and add_branch)
        factor = self.current_add_weight / self.base_add_weight
        if 'add_atom_inline' in adjusted_weights:
            adjusted_weights['add_atom_inline'] *= factor
        if 'add_branch' in adjusted_weights:
            adjusted_weights['add_branch'] *= factor
        
        # Renormalize weights to sum to 1
        total = sum(adjusted_weights.values())
        adjusted_weights = {k: v / total for k, v in adjusted_weights.items()}
        
        return adjusted_weights
    
    def get_atoms_to_add(self) -> int:
        """Get current number of atoms to add per mutation."""
        return self.current_atoms_per_add
    
    def get_statistics(self) -> Dict:
        """Get comprehensive statistics."""
        stats = self.detector.get_statistics()
        stats.update({
            'current_add_weight': self.current_add_weight,
            'current_atoms_per_add': self.current_atoms_per_add,
            'total_interventions': len(self.intervention_history),
            'intervention_history': self.intervention_history[-10:]  # Last 10 interventions
        })
        return stats
    
    def should_apply_intensive_mutation(self, stagnation_duration: int) -> bool:
        """
        Determine if intensive mutation should be applied.
        
        Args:
            stagnation_duration: Current stagnation duration
            
        Returns:
            True if intensive mutation should be applied
        """
        return stagnation_duration >= self.stagnation_threshold


# Integration helper functions

def integrate_with_generator(generator, adaptive_mutation: StagnationAdaptiveMutation):
    """
    Integrate adaptive mutation with MoleculeGenerator.
    
    This modifies the generator's mutation behavior to respond to stagnation.
    """
    # Store reference to adaptive mutation system
    generator.adaptive_mutation = adaptive_mutation
    
    # Override or extend the add_atoms mutation method
    original_add_atoms = generator.add_atoms
    
    def adaptive_add_atoms(smiles: str, n_atoms: int = None):
        """add_atoms that uses adaptive atom count."""
        if n_atoms is None:
            n_atoms = adaptive_mutation.get_atoms_to_add()
        return original_add_atoms(smiles, n_atoms)
    
    generator.add_atoms = adaptive_add_atoms
    
    logger.info("Stagnation-adaptive mutation with generator")


def update_generator_weights(generator, adaptive_mutation: StagnationAdaptiveMutation):
    """Update generator mutation weights based on current adaptive state."""
    adjusted_weights = adaptive_mutation.get_mutation_weights(generator.mutation_weights)
    generator.mutation_weights = adjusted_weights

# Multi-atom addition strategy

class MultiAtomAdditionMutation:
    """
   Apply multiple atom additions to the same molecule.
    
    When stagnation is detected, instead of adding more atoms in one mutation,
    apply the add_atoms mutation multiple times sequentially.
    """
    
    def __init__(self,
                 base_additions: int = 1,
                 max_additions: int = 5,
                 stagnation_threshold: int = 5,
                 additions_per_stagnation: float = 0.5):
        """
        Initialize multi-atom addition strategy.
        
        Args:
            base_additions: Base number of times to apply add_atoms
            max_additions: Maximum number of sequential additions
            stagnation_threshold: Generations to trigger adaptation
            additions_per_stagnation: Additional applications per stagnation period
        """
        self.base_additions = base_additions
        self.max_additions = max_additions
        self.stagnation_threshold = stagnation_threshold
        self.additions_per_stagnation = additions_per_stagnation
        
        self.detector = StagnationDetector(stagnation_threshold=stagnation_threshold)
        self.current_additions = base_additions
    
    def update(self, generation: int, best_fitness: float) -> Dict:
        """Update and return number of additions to apply."""
        is_stagnant, stagnation_duration = self.detector.update(generation, best_fitness)
        
        if is_stagnant:
            additional = int(stagnation_duration * self.additions_per_stagnation)
            self.current_additions = min(
                self.base_additions + additional,
                self.max_additions
            )
        else:
            self.current_additions = self.base_additions
        
        return {
            'is_stagnant': is_stagnant,
            'stagnation_duration': stagnation_duration,
            'additions_to_apply': self.current_additions
        }
    
    def apply_multiple_additions(self, generator, smiles: str) -> str:
        """
        Apply add_atoms mutation multiple times to the same molecule.
        
        Args:
            generator: MoleculeGenerator instance
            smiles: Starting SMILES string
            
        Returns:
            Modified SMILES after multiple additions
        """
        current_smiles = smiles
        
        for i in range(self.current_additions):
            try:
                new_smiles = generator.add_atoms(current_smiles, n_atoms=1)
                if new_smiles and generator.validate_molecule(new_smiles):
                    current_smiles = new_smiles
                    logger.debug(f"  Addition {i+1}/{self.current_additions} successful")
                else:
                    logger.debug(f"  Addition {i+1}/{self.current_additions} failed, keeping previous")
                    break
            except Exception as e:
                logger.debug(f"  Addition {i+1}/{self.current_additions} error: {e}")
                break
        
        return current_smiles


# Hybrid strategy combining both approaches

class HybridStagnationStrategy:
    """
    Combines weight boosting, atom count boosting, and multiple additions.
    """
    
    def __init__(self,
                 stagnation_threshold: int = 5,
                 base_add_weight: float = 1.0,
                 weight_boost_factor: float = 2.0,
                 base_atoms_per_add: int = 1,
                 max_atoms_per_add: int = 3,
                 base_addition_repeats: int = 1,
                 max_addition_repeats: int = 3):
        """Initialize hybrid strategy with all modes."""
        self.detector = StagnationDetector(stagnation_threshold=stagnation_threshold)
        
        # Weight boosting
        self.base_add_weight = base_add_weight
        self.weight_boost_factor = weight_boost_factor
        self.current_add_weight = base_add_weight
        
        # Atom count boosting
        self.base_atoms_per_add = base_atoms_per_add
        self.max_atoms_per_add = max_atoms_per_add
        self.current_atoms_per_add = base_atoms_per_add
        
        # Multiple additions
        self.base_addition_repeats = base_addition_repeats
        self.max_addition_repeats = max_addition_repeats
        self.current_addition_repeats = base_addition_repeats
    
    def update(self, generation: int, best_fitness: float) -> Dict:
        """Update all parameters based on stagnation."""
        is_stagnant, stagnation_duration = self.detector.update(generation, best_fitness)
        
        if is_stagnant:
            # Exponential weight boost
            weight_multiplier = self.weight_boost_factor ** (stagnation_duration / self.detector.stagnation_threshold)
            self.current_add_weight = self.base_add_weight * weight_multiplier
            
            # Linear atom count increase (not directly used, but logged for reference)
            self.current_atoms_per_add = min(
                self.base_atoms_per_add + stagnation_duration // 2,
                self.max_atoms_per_add
            )
            
            # Linear repeat increase (for repeated mutate_multiple calls)
            self.current_addition_repeats = min(
                self.base_addition_repeats + stagnation_duration // 3,
                self.max_addition_repeats
            )
            
            logger.info(f"HYBRID STAGNATION RESPONSE (duration: {stagnation_duration})")
            logger.info(f"   - Weight boost factor: {weight_multiplier:.2f} (applied to add_atom_inline and add_branch)")
            logger.info(f"   - Atoms per add: {self.current_atoms_per_add} (logged only)")
            logger.info(f"   - Repeats: {self.base_addition_repeats} → {self.current_addition_repeats}")
        else:
            self.current_add_weight = self.base_add_weight
            self.current_atoms_per_add = self.base_atoms_per_add
            self.current_addition_repeats = self.base_addition_repeats
        
        return {
            'is_stagnant': is_stagnant,
            'stagnation_duration': stagnation_duration,
            'add_weight': self.current_add_weight,
            'atoms_per_add': self.current_atoms_per_add,
            'addition_repeats': self.current_addition_repeats
        }
    
    def get_mutation_weights(self, base_weights: Dict[str, float]) -> Dict[str, float]:
        """
        Get adjusted mutation weights based on current stagnation state.
        
        Args:
            base_weights: Original mutation weights dictionary
            
        Returns:
            Adjusted mutation weights
        """
        adjusted_weights = base_weights.copy()
        
        # Apply weight boost to add_atom mutations (types 2 and 3: add_atom_inline and add_branch)
        factor = self.current_add_weight / self.base_add_weight
        if 'add_atom_inline' in adjusted_weights:
            adjusted_weights['add_atom_inline'] *= factor
        if 'add_branch' in adjusted_weights:
            adjusted_weights['add_branch'] *= factor
        
        # Renormalize weights to sum to 1
        total = sum(adjusted_weights.values())
        adjusted_weights = {k: v / total for k, v in adjusted_weights.items()}
        
        return adjusted_weights