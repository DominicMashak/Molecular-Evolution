
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'molev_utils')))

import archive as ar
import optimizer as op
import random

from molecule_generator import MoleculeGenerator

def main():
    # Example: Evolving 2D vectors where objective is sum and measures are the values
    random.seed(42)
    mutation_weights = [0.1] * 7 # Maybe change this
    generator = MoleculeGenerator(mutation_weights=mutation_weights)
    
    def generate_solution():
        """Generate one SMILES string."""
        return generator.generate_initial_population(1)
    
    def mutate_solution(parent):
        """Mutate one SMILES string."""
        return generator.mutate_multiple([parent])
    
    def evaluate_solution(solution):
        """Evaluate: objective is sum, measures are the vector elements."""
        return { # TODO: Replace this with calculations about the molecule. Start with num atoms and num bonds, then try more.
            'fitness': random.random(),
            'measure_0': random.random() * 10,
            'measure_1': random.random() * 10,
            'sum_squared': random.random() * 10  # Extra tracked property # Not that arbitrary extra properties can be stored.
        }
    
    # Create archive and optimizer
    archive = ar.MAPElitesArchive(
        # The number of measure dims should equal the number of measure keys. Not sure about specifying range yet.
        measure_dims=[10, 10],
        measure_keys=['measure_0', 'measure_1'],
        objective_key='fitness'
    )
    
    optimizer = op.MAPElitesOptimizer(
        archive=archive,
        generate_fn=generate_solution,
        mutate_fn=mutate_solution,
        evaluate_fn=evaluate_solution,
        random_init_size=50
    )
    
    # Run optimization
    history = optimizer.run(
        n_generations=100,
        iterations_per_generation=10,
        log_frequency=20
    )
    
    # Get best solution
    best = optimizer.get_best_solution()
    if best:
        print(f"\nBest solution: {best['solution']}")
        print(f"Properties: {best['properties']}")
    
    # Show some example solutions from different regions
    print("\nSample solutions from archive:")
    for entry in random.sample(optimizer.archive.get_all_solutions(), 
                               min(5, len(optimizer.archive))):
        print(f"  {entry['indices']}: {entry['solution']} "
              f"(fitness={entry['properties']['fitness']})")    
    #optimizer 
    
if __name__ == "__main__":
    main()