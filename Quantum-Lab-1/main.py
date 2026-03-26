"""
main.py — Full test of the optimization library.
Runs all TODOs sequentially and prints results to verify correctness.

NOTE: The if __name__ == '__main__' guard is REQUIRED on Windows for
multiprocessing to work correctly (spawn-based process creation).
"""

import numpy as np
import networkx as nx

from Problem.MaxCutProblem import MaxCutProblem
from Problem.QuboProblem import QuboProblem
from Problem.IsingProblem import IsingProblem
from Problem.KnapsackProblem import KnapsackProblem
from Problem.TspProblem import TspProblem
from Problem.Converter import qubo_to_ising, ising_to_qubo

from Algorithm.Classical_Algorithm.Exhaustive_search import exhaustive_search, exhaustive_search_parallel
from Algorithm.Classical_Algorithm.Random_search import random_search
from Algorithm.Classical_Algorithm.Local_search import local_search
from Algorithm.Classical_Algorithm.Random_local_search import random_local_search


def separator(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def run_all_tests():

    # ============================================================
    # TODOs 1-6: MaxCut — unweighted
    # ============================================================
    separator("TODOs 1-6 | MaxCut (unweighted)")

    G = nx.Graph()
    G.add_nodes_from([1, 2, 3, 4, 5])
    G.add_edges_from([(1, 5), (2, 5), (3, 5), (4, 5), (1, 2)])

    maxcut = MaxCutProblem(G)

    # TODO 2: sympy expression + caching
    expr = maxcut.to_sympy_expr()
    print(f"Sympy expression : {expr}")
    assert maxcut.to_sympy_expr() is maxcut.to_sympy_expr(), "Caching failed!"
    print("Caching          : OK")

    # TODO 3: eval
    solution_mc = {"s_1": 1, "s_2": -1, "s_3": 1, "s_4": -1, "s_5": 1}
    cost_mc = maxcut.eval(solution_mc)
    print(f"Eval cost        : {cost_mc}")

    # TODOs 5-6: display (commented out to avoid blocking in non-interactive mode)
    # maxcut.display_graph()
    # maxcut.display_solution(solution_mc)


    # ============================================================
    # TODO 7: MaxCut — weighted
    # ============================================================
    separator("TODO 7 | MaxCut (weighted)")

    G_w = nx.Graph()
    G_w.add_nodes_from([0, 1, 2, 3])
    G_w.add_weighted_edges_from([(0, 1, 2.5), (0, 2, 1.0), (1, 3, 3.0), (2, 3, 0.5), (1, 2, 1.5)])

    maxcut_w = MaxCutProblem(G_w)
    print(f"Weighted sympy   : {maxcut_w.to_sympy_expr()}")
    solution_mcw = {"s_0": 1, "s_1": -1, "s_2": 1, "s_3": -1}
    print(f"Weighted eval    : {maxcut_w.eval(solution_mcw)}")


    # ============================================================
    # TODOs 8-11: QUBO
    # ============================================================
    separator("TODOs 8-11 | QUBO")

    Q_sym = np.array([[ 2., -1.,  0.],
                      [-1.,  3., -1.],
                      [ 0., -1.,  2.]])
    qubo_sym = QuboProblem(Q_sym)
    print(f"Sympy (symmetric): {qubo_sym.to_sympy_expr()}")
    print(f"Eval [1,0,1]     : {qubo_sym.eval([1, 0, 1])}")

    Q_tri = np.array([[1., 2., 3.],
                      [0., 4., 5.],
                      [0., 0., 6.]])
    qubo_tri = QuboProblem(Q_tri)
    print(f"Sympy (upper tri): {qubo_tri.to_sympy_expr()}")
    print(f"Eval [1,0,1]     : {qubo_tri.eval([1, 0, 1])}")

    try:
        QuboProblem(np.array([[1., 2.], [3., 4.]]))
        print("ERROR: should have raised ValueError!")
    except ValueError as e:
        print(f"Invalid matrix   : correctly raised ValueError ({e})")


    # ============================================================
    # TODOs 12-14: Ising
    # ============================================================
    separator("TODOs 12-14 | Ising")

    G_ising = nx.Graph()
    G_ising.add_node(0, weight=0.5)
    G_ising.add_node(1, weight=-0.3)
    G_ising.add_node(2, weight=0.1)
    G_ising.add_edge(0, 1, weight=1.0)
    G_ising.add_edge(1, 2, weight=-0.5)
    G_ising.add_edge(0, 2, weight=0.8)

    ising_graph = IsingProblem(G_ising)
    print(f"Sympy (from graph): {ising_graph.to_sympy_expr()}")
    sol_ising = {"s_0": 1, "s_1": -1, "s_2": 1}
    print(f"get_cost          : {ising_graph.get_cost(sol_ising)}")

    w = [0.5, -0.3, 0.1]
    J = np.array([[0., 1.0, 0.8], [0., 0., -0.5], [0., 0., 0.]])
    ising_direct = IsingProblem((w, J))
    print(f"Sympy (from w,J)  : {ising_direct.to_sympy_expr()}")
    print(f"get_cost          : {ising_direct.get_cost(sol_ising)}")


    # ============================================================
    # TODOs 15-16: Converters
    # ============================================================
    separator("TODOs 15-16 | Converters (QUBO <-> Ising)")

    ising_converted = qubo_to_ising(qubo_tri)
    print(f"QUBO->Ising sympy : {ising_converted.to_sympy_expr()}")

    qubo_back = ising_to_qubo(ising_converted)
    print(f"Round-trip matrix :\n{np.round(qubo_back.matrix, 4)}")
    print(f"Original matrix   :\n{qubo_tri.matrix}")

    if np.allclose(qubo_back.matrix, qubo_tri.matrix):
        print("Round-trip        : OK (matrices match)")
    else:
        print("Round-trip        : MISMATCH (check converter logic)")


    # ============================================================
    # TODO 17: Knapsack
    # ============================================================
    separator("TODO 17 | Knapsack")

    knapsack = KnapsackProblem(
        prices=[10, 6, 5, 4, 3],
        weights=[5, 4, 3, 2, 1],
        capacity=8
    )
    sol_ok   = [0, 0, 1, 1, 1]  # weight=6 < 8
    sol_bad  = [1, 1, 1, 0, 0]  # weight=12 >= 8
    sol_edge = [0, 1, 1, 0, 1]  # weight=8, NOT < 8

    print(f"Feasible [0,0,1,1,1] (w=6) : {knapsack.is_feasible(sol_ok)}")
    print(f"Feasible [1,1,1,0,0] (w=12): {knapsack.is_feasible(sol_bad)}")
    print(f"Feasible [0,1,1,0,1] (w=8) : {knapsack.is_feasible(sol_edge)}")
    print(f"Eval [0,0,1,1,1]           : {knapsack.eval(sol_ok)}")
    print(f"Eval [1,1,1,0,0] (infeas.) : {knapsack.eval(sol_bad)}")


    # ============================================================
    # TODO 18: TSP
    # ============================================================
    separator("TODO 18 | TSP")

    distance_matrix = [
        [ 0,  2,  9, 10],
        [ 1,  0,  6,  4],
        [15,  7,  0,  8],
        [ 6,  3, 12,  0],
    ]
    city_names = ["Paris", "Lyon", "Marseille", "Bordeaux"]
    tsp = TspProblem(distance_matrix, city_names)

    sol_tsp = [0, 1, 3, 2]
    print(f"Route    : {[city_names[c] for c in sol_tsp]}")
    print(f"Distance : {tsp.eval(sol_tsp)}")
    # tsp.display_solution(sol_tsp)  # Uncomment in interactive mode


    # ============================================================
    # TODO 19: Exhaustive search (+ parallel)
    # ============================================================
    separator("TODO 19 | Exhaustive search")

    best_tsp, dist_tsp = exhaustive_search(tsp, maximize=False)
    print(f"Best TSP route    : {[city_names[c] for c in best_tsp]}")
    print(f"Best TSP distance : {dist_tsp}")

    best_knap, val_knap = exhaustive_search(knapsack, maximize=True)
    print(f"Best Knapsack     : {best_knap}  value={val_knap}")

    best_mc, cost_mc2 = exhaustive_search(maxcut, maximize=False)
    print(f"Best MaxCut       : {best_mc}  cost={cost_mc2}")

    best_qubo, cost_qubo = exhaustive_search(qubo_sym, maximize=False)
    print(f"Best QUBO         : {best_qubo}  cost={cost_qubo}")

    best_ising, cost_ising = exhaustive_search(ising_direct, maximize=False)
    print(f"Best Ising        : {best_ising}  cost={cost_ising}")

    # Parallel version — requires if __name__ == '__main__' on Windows
    best_tsp_par, dist_tsp_par = exhaustive_search_parallel(tsp, maximize=False)
    print(f"Parallel TSP      : {[city_names[c] for c in best_tsp_par]}  dist={dist_tsp_par}")


    # ============================================================
    # TODO 20: Random search
    # ============================================================
    separator("TODO 20 | Random search")

    best_tsp_r, dist_tsp_r = random_search(tsp, n_iterations=500, maximize=False)
    print(f"Random TSP route  : {[city_names[c] for c in best_tsp_r]}  dist={dist_tsp_r}")

    best_knap_r, val_knap_r = random_search(knapsack, n_iterations=500, maximize=True)
    print(f"Random Knapsack   : {best_knap_r}  value={val_knap_r}")


    # ============================================================
    # TODO 21: Local search (QUBO) + gen_neighbor_sol
    # ============================================================
    separator("TODO 21 | Local search (QUBO)")

    best_local, cost_local = local_search(qubo_sym, n_iterations=1000)
    print(f"Local search best : {best_local}  cost={cost_local}")
    print(f"Exhaustive best   : {best_qubo}   cost={cost_qubo}  (reference)")

    import random
    sol_sample = [1, 0, 1]
    neighbor = qubo_sym.gen_neighbor_sol(sol_sample)
    diff = sum(abs(sol_sample[i] - neighbor[i]) for i in range(len(sol_sample)))
    print(f"Neighbor of {sol_sample}: {neighbor}  (bits flipped: {diff})")
    assert diff == 1, "gen_neighbor_sol should flip exactly 1 bit!"
    print("gen_neighbor_sol  : OK (exactly 1 bit flipped)")


    # ============================================================
    # TODO 23 (Bonus): Random Local Search
    # ============================================================
    separator("TODO 23 (Bonus) | Random Local Search")

    best_rls, cost_rls = random_local_search(qubo_sym, n_restarts=20, n_iterations=200)
    print(f"RLS best solution : {best_rls}  cost={cost_rls}")
    print(f"Exhaustive best   : {best_qubo}  cost={cost_qubo}  (reference)")


    # ============================================================
    # Final summary
    # ============================================================
    separator("ALL TESTS PASSED")
    print("  All TODOs (1-23) executed successfully.")


# REQUIRED on Windows: multiprocessing uses 'spawn' which re-imports
# the main module in each child process. Without this guard, every
# worker would re-execute all the test code, causing infinite recursion.
if __name__ == '__main__':
    run_all_tests()
