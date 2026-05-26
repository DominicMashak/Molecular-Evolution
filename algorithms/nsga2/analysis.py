def analyze_parent_child_performance(self, parents, children):
    """
    Analyze the performance of parent and child populations in terms of beta values and atom counts.

    This method computes the average beta (objective 0) and average atom counts (objective 1) for the given
    parent and child populations. It then calculates the improvement in beta and the change in atom counts
    between parents and children. The results are appended to the parent_child_stats list as a dictionary
    containing generation information and the computed averages and differences.

    Parameters:
        parents (list): A list of parent objects, each potentially having 'beta_surrogate', 'objectives', and 'natoms' attributes.
        children (list): A list of child objects, each potentially having 'beta_surrogate', 'objectives', and 'natoms' attributes.

    Returns:
        None: This method does not return a value; it modifies the instance's parent_child_stats list.

    Raises:
        None: Exceptions during attribute access are caught and default values (0.0 or 0) are used.
    """
    if not parents or not children:
        return

    # Compute average beta (objective 0) and atom counts (objective 1) for parents and children
    try:
        avg_parent_beta = sum(getattr(p, 'beta_surrogate', (p.objectives[0] if p.objectives else 0.0)) for p in parents) / len(parents)
    except Exception:
        avg_parent_beta = 0.0
    try:
        avg_child_beta = sum(getattr(c, 'beta_surrogate', (c.objectives[0] if c.objectives else 0.0)) for c in children) / len(children)
    except Exception:
        avg_child_beta = 0.0

    try:
        avg_parent_atoms = sum(getattr(p, 'natoms', (int(p.objectives[1]) if len(p.objectives) > 1 else 0)) for p in parents) / len(parents)
    except Exception:
        avg_parent_atoms = 0.0
    try:
        avg_child_atoms = sum(getattr(c, 'natoms', (int(c.objectives[1]) if len(c.objectives) > 1 else 0)) for c in children) / len(children)
    except Exception:
        avg_child_atoms = 0.0

    self.parent_child_stats.append({
        'generation': getattr(self, 'generation', None),
        'avg_parent_beta': avg_parent_beta,
        'avg_child_beta': avg_child_beta,
        'avg_parent_atoms': avg_parent_atoms,
        'avg_child_atoms': avg_child_atoms,
        'beta_improvement': avg_child_beta - avg_parent_beta,
        'atom_change': avg_child_atoms - avg_parent_atoms
    })
