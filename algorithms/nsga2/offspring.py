import random

def create_offspring(self, parents):
    """
    Generates a list of offspring individuals from a given set of parent individuals using tournament selection and mutation.

    The method repeatedly selects a subset of parents via tournament selection, chooses the winner based on rank and crowding distance,
    applies mutation to the winner's SMILES string, validates the mutated molecule, and creates a new individual if valid.
    The process continues until the desired number of children is produced.

    Args:
        parents (list): A list of parent individuals to select from.

    Returns:
        list: A list of newly created offspring individuals.
    """
    children = []
    while len(children) < self.n_children:
        tournament_size = 3
        tournament = random.sample(parents, tournament_size)
        winner = min(tournament, key=lambda x: (x.rank, -x.crowding_distance))
        mutated_smiles = self.generator.mutate_multiple(winner.smiles)
        if mutated_smiles and self.generator.validate_molecule(mutated_smiles):
            child = self.create_individual(mutated_smiles, self.generation + 1)
            children.append(child)
    return children
