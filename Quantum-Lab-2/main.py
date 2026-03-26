"""
main.py — Test file for Lab Session 2.
Runs all the new algorithms (SA, B&B, Gradient Descent) to verify they work.
"""
import numpy as np
import random
from Problem.QuboProblem import QuboProblem
from Problem.KnapsackProblem import KnapsackProblem
from Problem.KnapsackVolumetric import KnapsackVolumetric
from Problem.TspProblem import TspProblem
from Problem.Converter import knapsack_to_qubo
from Algorithm.Classical_Algorithm.Simulated_annealing import simulated_annealing, simulated_annealing_tsp
from Algorithm.Classical_Algorithm.Branch_and_bound import BranchAndBound
from Algorithm.Classical_Algorithm.Gradient_descent import gradient_descent, numerical_gradient
from Algorithm.Classical_Algorithm.Exhaustive_search import exhaustive_search

def separator(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

def run_all_tests():
    # TODOs 1-3: Simulated Annealing on QUBO
    separator("TODOs 1-3 | Simulated Annealing (QUBO)")
    Q_sym = np.array([[ 2., -1.,  0.],
                      [-1.,  3., -1.],
                      [ 0., -1.,  2.]])
    qubo = QuboProblem(Q_sym)
    best_exhaustive, cost_exhaustive = exhaustive_search(qubo, maximize=False)
    print(f"Exhaustive best : {best_exhaustive} cost={cost_exhaustive}")
    best_sa, cost_sa = simulated_annealing(qubo, initial_temperature=50.0,
                                            decreasing_factor=0.95,
                                            steps_per_temp=20,
                                            temp_threshold=0.001)
    print(f"SA best : {best_sa} cost={cost_sa}")
    # Test different cooling schedules
    for schedule in ["exponential", "linear", "geometric"]:
        best, cost = simulated_annealing(qubo, initial_temperature=50.0,
                                          decreasing_factor=0.9,
                                          steps_per_temp=15,
                                          temp_threshold=0.01,
                                          cooling_schedule=schedule)
        print(f"SA ({schedule}) : {best} cost={cost}")

    # TODO 6: Branch and Bound on Knapsack
    separator("TODO 6 | Branch and Bound (Knapsack)")
    knapsack = KnapsackProblem(prices=[10, 6, 5, 4, 3],
                                weights=[5, 4, 3, 2, 1],
                                capacity=8)
    best_knap_ex, val_knap_ex = exhaustive_search(knapsack, maximize=True)
    print(f"Exhaustive best : {best_knap_ex} value={val_knap_ex}")
    bb = BranchAndBound(knapsack)
    best_bb, val_bb = bb.solve()
    print(f"B&B best : {best_bb} value={val_bb}")
    if val_bb == val_knap_ex:
        print("B&B : OK (matches exhaustive)")
    else:
        print("B&B : MISMATCH!")

    # TODO 7: Knapsack to QUBO conversion
    separator("TODO 7 | Knapsack to QUBO")
    qubo_from_knapsack = knapsack_to_qubo(knapsack)
    best_qubo_knap, cost_qubo_knap = exhaustive_search(qubo_from_knapsack, maximize=False)
    print(f"QUBO optimal sol : {best_qubo_knap} cost={cost_qubo_knap}")
    print(f"Knapsack optimal : {best_knap_ex} value={val_knap_ex}")
    if best_qubo_knap == best_knap_ex:
        print("Conversion : OK (optimal preserved)")
    else:
        print("Conversion : Check penalty parameter")

    # TODO 8: Gradient Descent
    separator("TODO 8 | Gradient Descent")
    def f_convex(x, y):
        return x**2 + y**2
    def grad_convex(x, y):
        return 2*x, 2*y
    x_hist, y_hist, f_hist = gradient_descent(f_convex, grad_convex,
                                               x0=4.0, y0=3.0,
                                               learning_rate=0.1,
                                               n_iterations=100)
    print(f"Convex start : ({x_hist[0]:.4f}, {y_hist[0]:.4f}), f={f_hist[0]:.4f}")
    print(f"Convex final : ({x_hist[-1]:.6f}, {y_hist[-1]:.6f}), f={f_hist[-1]:.6f}")
    print(f"Iterations : {len(x_hist) - 1}")
    # Non-convex function
    def f_nonconvex(x, y):
        return x**2 + y**2 + 10*(2 - np.cos(2*np.pi*x) - np.cos(2*np.pi*y))
    def grad_nonconvex(x, y):
        return 2*x + 20*np.pi*np.sin(2*np.pi*x), 2*y + 20*np.pi*np.sin(2*np.pi*y)
    x_hist2, y_hist2, f_hist2 = gradient_descent(f_nonconvex, grad_nonconvex,
                                                   x0=3.5, y0=2.5,
                                                   learning_rate=0.001,
                                                   n_iterations=2000)
    print(f"Non-convex start : ({x_hist2[0]:.4f}, {y_hist2[0]:.4f}), f={f_hist2[0]:.4f}")
    print(f"Non-convex final : ({x_hist2[-1]:.6f}, {y_hist2[-1]:.6f}), f={f_hist2[-1]:.6f}")
    # Test numerical gradient
    gx_a, gy_a = grad_convex(2.0, 1.5)
    gx_n, gy_n = numerical_gradient(f_convex, 2.0, 1.5)
    print(f"Analytical grad : ({gx_a}, {gy_a})")
    print(f"Numerical grad : ({gx_n:.6f}, {gy_n:.6f})")

    # TODO 9: SA for TSP
    separator("TODO 9 | SA for TSP")
    distance_matrix = [
        [ 0,  2,  9, 10],
        [ 1,  0,  6,  4],
        [15,  7,  0,  8],
        [ 6,  3, 12,  0],
    ]
    city_names = ["Paris", "Lyon", "Marseille", "Bordeaux"]
    tsp = TspProblem(distance_matrix, city_names)
    best_tsp_ex, dist_tsp_ex = exhaustive_search(tsp, maximize=False)
    print(f"Exhaustive best : {[city_names[c] for c in best_tsp_ex]} dist={dist_tsp_ex}")
    best_tsp_sa, dist_tsp_sa = simulated_annealing_tsp(tsp, initial_temperature=500.0,
                                                        decreasing_factor=0.99,
                                                        steps_per_temp=30,
                                                        temp_threshold=0.01)
    print(f"SA TSP best : {[city_names[c] for c in best_tsp_sa]} dist={dist_tsp_sa}")

    # TODO 10: KnapsackVolumetric
    separator("TODO 10 | KnapsackVolumetric (Bonus)")
    kv = KnapsackVolumetric(prices=[10, 6, 5, 4, 3],
                             weights=[5, 4, 3, 2, 1],
                             capacity=8,
                             volumes=[3, 2, 4, 1, 2],
                             max_volume=6)
    best_kv_ex, val_kv_ex = exhaustive_search(kv, maximize=True)
    print(f"Exhaustive best : {best_kv_ex} value={val_kv_ex}")
    bb_vol = BranchAndBound(kv)
    best_bb_vol, val_bb_vol = bb_vol.solve()
    print(f"B&B best : {best_bb_vol} value={val_bb_vol}")
    if val_bb_vol == val_kv_ex:
        print("B&B Volumetric : OK (matches exhaustive)")
    else:
        print("B&B Volumetric : MISMATCH!")
    # Test that zero volumes = regular knapsack
    kv_zero = KnapsackVolumetric(prices=[10, 6, 5, 4, 3],
                                  weights=[5, 4, 3, 2, 1],
                                  capacity=8)
    bb_zero = BranchAndBound(kv_zero)
    _, val_zero = bb_zero.solve()
    print(f"Zero volumes B&B : value={val_zero} (should match regular knapsack: {val_knap_ex})")

    separator("ALL TESTS COMPLETED")
    print("All Lab Session 2 TODOs executed successfully.")

if __name__ == '__main__':
    run_all_tests()
