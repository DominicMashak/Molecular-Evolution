def dominates(ind1, ind2, optimize_objectives=None):
    """
    Determines whether one individual dominates another in multi-objective optimization.
    An individual `ind1` is said to dominate `ind2` if it is at least as good in all objectives
    and strictly better in at least one objective, according to the specified optimization criteria.
    Args:
        ind1: An object with an `objectives` attribute (list of objective values).
        ind2: An object with an `objectives` attribute (list of objective values).
        optimize_objectives (list of tuple, optional): A list specifying the optimization type for each objective.
            Each tuple is of the form (`type`, `target`), where `type` is one of:
                - 'max': Objective should be maximized.
                - 'min': Objective should be minimized.
                - 'target': Objective should be as close as possible to `target`.
            If None, all objectives are maximized by default.
    Returns:
        bool: True if `ind1` dominates `ind2`, False otherwise.
    Raises:
        ValueError: If an unknown optimization type is specified in `optimize_objectives`.
    """
    if optimize_objectives is None:
        optimize_objectives = [('max', None)] * len(ind1.objectives)  # Default
    
    better_or_equal = True
    strictly_better = False
    
    for i, (obj1, obj2) in enumerate(zip(ind1.objectives, ind2.objectives)):
        opt_type, target = optimize_objectives[i]
        if opt_type == 'max':
            # Maximize: ind1 better if obj1 >= obj2
            if obj1 < obj2:
                better_or_equal = False
            if obj1 > obj2:
                strictly_better = True
        elif opt_type == 'min':
            # Minimize: ind1 better if obj1 <= obj2
            if obj1 > obj2:
                better_or_equal = False
            if obj1 < obj2:
                strictly_better = True
        elif opt_type == 'target':
            # Target: minimize |obj - target|
            dist1 = abs(obj1 - target)
            dist2 = abs(obj2 - target)
            if dist1 > dist2:
                better_or_equal = False
            if dist1 < dist2:
                strictly_better = True
        else:
            raise ValueError(f"Unknown optimize type: {opt_type}")
    
    return better_or_equal and strictly_better

def fast_non_dominated_sort(population, optimize_objectives=None):
    fronts = [[]]
    for p in population:
        p.dominated_by = 0
        p.dominates = []
        for q in population:
            if p is q:
                continue
            if dominates(p, q, optimize_objectives):
                p.dominates.append(q)
            elif dominates(q, p, optimize_objectives):
                p.dominated_by += 1
        if p.dominated_by == 0:
            p.rank = 0
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in p.dominates:
                q.dominated_by -= 1
                if q.dominated_by == 0:
                    q.rank = i + 1
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    if not fronts[-1]:
        fronts.pop()
    return fronts