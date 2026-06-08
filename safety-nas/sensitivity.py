import numpy as np
import pulp


def heuristic3(sensitivity, budget, tradeoff_map):
    """
    sensitivity : list or np.array, shape (n_states,)
    budget      : float
    tradeoff_map: np.array shape (n_levels, 2)

    Column 0 = error
    Column 1 = cost

    Returns:
        selection: list of selected level indices (0-based)
    """

    sensitivity = np.asarray(sensitivity, dtype=float)
    tradeoff_map = np.asarray(tradeoff_map, dtype=float)

    n_states = len(sensitivity)
    n_levels = tradeoff_map.shape[0]

    model = pulp.LpProblem("heuristic3", pulp.LpMinimize)

    # Binary selection variables
    a = pulp.LpVariable.dicts(
        "a",
        ((i, l) for i in range(n_states) for l in range(n_levels)),
        cat="Binary"
    )

    # Objective
    model += pulp.lpSum(
        a[(i, l)]
        * tradeoff_map[l, 0]
        * sensitivity[i]
        for i in range(n_states)
        for l in range(n_levels)
    )

    # Budget constraint
    model += pulp.lpSum(
        a[(i, l)]
        * tradeoff_map[l, 1]
        for i in range(n_states)
        for l in range(n_levels)
    ) <= budget

    # Exactly one level per state
    for i in range(n_states):
        model += pulp.lpSum(
            a[(i, l)]
            for l in range(n_levels)
        ) == 1

    model.solve(pulp.PULP_CBC_CMD(msg=False))

    selection = []

    for i in range(n_states):
        chosen = max(
            range(n_levels),
            key=lambda l: pulp.value(a[(i, l)])
        )
        selection.append(chosen)

    return selection