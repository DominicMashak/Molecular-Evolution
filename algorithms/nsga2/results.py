def save_results(self):
    """
    Saves the results of the NSGA-II optimization process to JSON files.

    This method performs a fast non-dominated sort on the current population to identify the Pareto front.
    It then extracts the SMILES strings, objectives, and generation numbers for individuals on the first front
    and saves them to 'pareto_front_molecules.json'. Additionally, it saves the parent-child statistics to
    'parent_child_stats.json', the final mutation weights to 'final_mutation_weights.json', and calls
    save_molecule_database to persist the molecule database.

    Returns:
        None
    """
    fronts = self.fast_non_dominated_sort(self.population, self.minimize_objectives)
    results = []
    for ind in fronts[0] if fronts else []:
        results.append({
            'smiles': ind.smiles,
            'objectives': ind.objectives,
            'generation': ind.generation
        })
    pareto_file = self.output_dir / "pareto_front_molecules.json"
    with open(pareto_file, 'w') as f:
        import json
        json.dump(results, f, indent=2)
    stats_file = self.output_dir / "parent_child_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(self.parent_child_stats, f, indent=2)
    weights_file = self.output_dir / "final_mutation_weights.json"
    with open(weights_file, 'w') as f:
        json.dump(self.generator.mutation_weights, f, indent=2)
    self.save_molecule_database()