import random


def local_search(problem, n_iterations=1000, maximize=False):
    """
    Local search starting from a random initial solution.
    At each step, moves to a neighbor if it improves the cost.

    Compatible with any problem that implements:
        - generate_random_solution()
        - gen_neighbor_sol(solution)
        - eval(solution)

    :param problem:      Problem instance.
    :param n_iterations: Maximum number of iterations.
    :param maximize:     If True, maximize cost; if False, minimize cost.
    :return: (best_solution, best_cost) tuple.
    """
    current_solution = problem.generate_random_solution()
    current_cost     = problem.eval(current_solution)

    best_solution = current_solution if not isinstance(current_solution, list) else current_solution.copy()
    best_cost     = current_cost

    for _ in range(n_iterations):
        neighbor      = problem.gen_neighbor_sol(current_solution)
        neighbor_cost = problem.eval(neighbor)

        # Accept the neighbor if it improves the objective
        improved = (neighbor_cost > current_cost) if maximize else (neighbor_cost < current_cost)
        if improved:
            current_solution = neighbor
            current_cost     = neighbor_cost

            better_than_best = (current_cost > best_cost) if maximize else (current_cost < best_cost)
            if better_than_best:
                best_cost     = current_cost
                best_solution = current_solution if not isinstance(current_solution, list) else current_solution.copy()

    return best_solution, best_cost
