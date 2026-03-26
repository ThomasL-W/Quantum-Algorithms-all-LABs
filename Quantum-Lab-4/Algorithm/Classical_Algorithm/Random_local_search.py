import random


def random_local_search(problem, n_restarts=10, n_iterations=200, maximize=False):
    """
    Random Local Search (RLS) — combines random search and local search.

    Strategy:
        1. Pick a random starting solution (random search phase).
        2. Apply local search from that starting point.
        3. Repeat for n_restarts independent runs.
        4. Return the best solution found across all runs.

    Compatible with any problem that implements:
        - generate_random_solution()
        - gen_neighbor_sol(solution)
        - eval(solution)

    :param problem:      Problem instance.
    :param n_restarts:   Number of independent random restarts.
    :param n_iterations: Number of local search steps per restart.
    :param maximize:     If True, maximize cost; if False, minimize cost.
    :return: (best_solution, best_cost) tuple.
    """
    best_solution = None
    best_cost = float("-inf") if maximize else float("inf")

    for restart in range(n_restarts):
        # --- Random phase: pick a random starting solution ---
        current_solution = problem.generate_random_solution()
        current_cost     = problem.eval(current_solution)

        # --- Local search phase: greedy descent from this starting point ---
        for _ in range(n_iterations):
            neighbor      = problem.gen_neighbor_sol(current_solution)
            neighbor_cost = problem.eval(neighbor)

            improved = (neighbor_cost > current_cost) if maximize else (neighbor_cost < current_cost)
            if improved:
                current_solution = neighbor
                current_cost     = neighbor_cost

        # --- Update global best across all restarts ---
        better_than_best = (current_cost > best_cost) if maximize else (current_cost < best_cost)
        if better_than_best:
            best_cost     = current_cost
            best_solution = current_solution if not isinstance(current_solution, list) else current_solution.copy()

    return best_solution, best_cost
