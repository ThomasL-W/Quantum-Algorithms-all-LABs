import random


def random_search(problem, n_iterations=1000, maximize=False):
    """
    Randomly samples configurations and tracks the best.

    Instead of generating the complete search space (which grows exponentially
    as 2^n or n!), this version generates random solutions on-the-fly using
    problem.generate_random_solution(), making it scalable to large instances.

    :param problem:      Problem instance with generate_random_solution() and eval().
    :param n_iterations: Number of random configurations to try.
    :param maximize:     If True, maximize cost; if False, minimize cost.
    :return: (best_solution, best_cost) tuple.
    """
    best_solution = None
    best_cost = float("-inf") if maximize else float("inf")

    for _ in range(n_iterations):
        solution = problem.generate_random_solution()
        cost = problem.eval(solution)

        if maximize and cost > best_cost:
            best_cost, best_solution = cost, solution
        elif not maximize and cost < best_cost:
            best_cost, best_solution = cost, solution

    return best_solution, best_cost
