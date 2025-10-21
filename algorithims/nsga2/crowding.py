def crowding_distance(front, optimize_objectives=None):
    """
    Calculate the crowding distance for each individual in a Pareto front.
    This function implements the crowding distance assignment used in the NSGA-II algorithm.
    It assigns infinite crowding distance to the boundary individuals (first and last in sorted order)
    and computes the crowding distance for others based on the normalized difference in objective values.
    Parameters:
        front (list): A list of individuals, where each individual has an 'objectives' attribute
                      (a list of float values representing objective function values) and a
                      'crowding_distance' attribute that will be set by this function.
        optimize_objectives (list of tuples, optional): A list specifying the optimization type for each objective.
                                                        Each tuple is (opt_type, target), where opt_type is 'max', 'min', or 'target',
                                                        and target is a float value used for 'target' optimization.
                                                        If None, defaults to ('max', None) for all objectives.
    Notes:
        - For fronts with 2 or fewer individuals, all are assigned infinite crowding distance.
        - The front is sorted in-place for each objective.
        - Crowding distance is accumulated across all objectives.
    """
    n = len(front)
    if n <= 2:
        for ind in front:
            ind.crowding_distance = float('inf')
        return
    
    for ind in front:
        ind.crowding_distance = 0
    
    n_objectives = len(front[0].objectives)
    for obj_idx in range(n_objectives):
        opt_type, target = optimize_objectives[obj_idx] if optimize_objectives else ('max', None)
        
        # Sort by this objective
        if opt_type == 'max':
            front.sort(key=lambda x: -x.objectives[obj_idx])  # Descending
        elif opt_type == 'min':
            front.sort(key=lambda x: x.objectives[obj_idx])  # Ascending
        elif opt_type == 'target':
            front.sort(key=lambda x: abs(x.objectives[obj_idx] - target))  # Ascending distance
        
        obj_values = [ind.objectives[obj_idx] for ind in front]
        obj_range = max(obj_values) - min(obj_values)
        if obj_range == 0:
            continue  # No variation, skip
        
        front[0].crowding_distance = float('inf')
        front[-1].crowding_distance = float('inf')
        for i in range(1, n - 1):
            contribution = (front[i + 1].objectives[obj_idx] - front[i - 1].objectives[obj_idx]) / obj_range
            front[i].crowding_distance += contribution
