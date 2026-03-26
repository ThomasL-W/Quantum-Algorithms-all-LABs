import numpy as np
from concurrent.futures import ProcessPoolExecutor
import multiprocessing


def exhaustive_search(problem, maximize=False):
    """
    Tries every solution in the complete search space and returns the best one.

    Compatible with: KnapsackProblem, TspProblem, IsingProblem,
                     QuboProblem, MaxCutProblem.

    :param problem:  Problem instance with generate_complete_search_space() and eval().
    :param maximize: If True, maximize cost; if False, minimize cost.
    :return: (best_solution, best_cost) tuple.
    """
    search_space = problem.generate_complete_search_space()

    best_solution = None
    best_cost = float("-inf") if maximize else float("inf")

    for solution in search_space:
        cost = problem.eval(solution)
        if maximize and cost > best_cost:
            best_cost, best_solution = cost, solution
        elif not maximize and cost < best_cost:
            best_cost, best_solution = cost, solution

    return best_solution, best_cost


def exhaustive_search_parallel(problem, maximize=False):
    """
    Parallelized exhaustive search using all available CPU cores.

    :param problem:  Problem instance with generate_complete_search_space() and eval().
    :param maximize: If True, maximize cost; if False, minimize cost.
    :return: (best_solution, best_cost) tuple.
    """
    search_space = problem.generate_complete_search_space()
    n_cores = multiprocessing.cpu_count()

    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        costs = list(executor.map(problem.eval, search_space))

    best_idx = int(np.argmax(costs)) if maximize else int(np.argmin(costs))
    return search_space[best_idx], costs[best_idx]
