
import archive as ar
import optimizer as op
import random

def main():
    # Example: Evolving 2D vectors where objective is sum and measures are the values
    random.seed(42)
    
    def generate_solution():
        """Generate a random 2D vector with values in [0, 9]."""
        return [random.randint(0, 9), random.randint(0, 9)]
    
    def mutate_solution(parent):
        """Mutate by randomly changing one element by ±1."""
        child = parent.copy()
        idx = random.randint(0, 1)
        child[idx] = max(0, min(9, child[idx] + random.choice([-1, 1])))
        return child
    
    def evaluate_solution(solution):
        """Evaluate: objective is sum, measures are the vector elements."""
        return {
            'fitness': sum(solution),
            'measure_0': solution[0],
            'measure_1': solution[1],
            'sum_squared': sum(x**2 for x in solution)  # Extra tracked property
        }
    
    # Create archive and optimizer
    archive = ar.MAPElitesArchive(
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