"""
main.py — Test file for Lab Session 3.
Tests the D-Wave quantum annealing simulator.
"""
import numpy as np
import networkx as nx
from Problem.IsingProblem import IsingProblem
from Algorithm.Classical_Algorithm.Exhaustive_search import exhaustive_search
from quantum_solver.dwave_simulator import DwaveSimulator, create_ising_instance_random

def separator(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

def run_all_tests():
    # TODO 1-3 : Build DwaveSimulator, H_init, H_final
    separator("TODOs 1-3 | DwaveSimulator, H_init, H_final")
    G = nx.Graph()
    G.add_node(0, weight=0.5)
    G.add_node(1, weight=-0.3)
    G.add_node(2, weight=0.1)
    G.add_edge(0, 1, weight=1.0)
    G.add_edge(1, 2, weight=-0.5)
    G.add_edge(0, 2, weight=0.8)
    ising = IsingProblem(G)
    sim = DwaveSimulator()
    H_init = sim.build_Hinit(ising.n)
    H_final = sim.build_Hfinal(ising)
    print(f"Number of qubits : {ising.n}")
    print(f"H_init shape : {H_init.shape}")
    print(f"H_final shape : {H_final.shape}")
    print(f"Annealing schedule : {len(sim.annealing_schedule)} steps")
    print(f"A(0)={sim.annealing_schedule[0][0]}, B(0)={sim.annealing_schedule[0][1]}")
    print(f"A(end)={sim.annealing_schedule[-1][0]}, B(end)={sim.annealing_schedule[-1][1]}")

    # TODO 4 : Simulate evolution
    separator("TODO 4 | Simulate evolution")
    all_eigenvalues, all_eigenvectors = sim.simulate_evolution(ising, nb_eigenvalues=5)
    print(f"Number of steps : {len(all_eigenvalues)}")
    print(f"Eigenvalues at step 0 : {all_eigenvalues[0]}")
    print(f"Eigenvalues at last step : {all_eigenvalues[-1]}")
    # Check vs exhaustive search
    best_sol, best_cost = exhaustive_search(ising, maximize=False)
    final_lowest = all_eigenvalues[-1][0]
    print(f"Lowest eigenvalue at end : {final_lowest:.6f}")
    print(f"Exhaustive search optimal cost : {best_cost}")
    if abs(final_lowest - best_cost) < 1e-5:
        print("Check : OK (eigenvalue matches optimal cost)")
    else:
        print("Check : MISMATCH")
    # Check eigenvector
    ground_state = all_eigenvectors[-1][:, 0]
    # The eigenvector index with max amplitude gives the binary solution
    max_idx = np.argmax(np.abs(ground_state))
    n = ising.n
    binary_str = format(max_idx, f'0{n}b')
    # Convert binary to spin: 0 -> +1, 1 -> -1
    spin_solution = {}
    for i in range(n):
        if binary_str[i] == '0':
            spin_solution[f"s_{ising.nodes[i]}"] = 1
        else:
            spin_solution[f"s_{ising.nodes[i]}"] = -1
    print(f"Ground state index : {max_idx} (binary: {binary_str})")
    print(f"Spin solution from eigenvector : {spin_solution}")
    print(f"Exhaustive search solution : {best_sol}")

    # TODO 5 : Spectral gap
    separator("TODO 5 | Spectral gap")
    gaps = []
    for step in range(len(all_eigenvalues)):
        gap = all_eigenvalues[step][1] - all_eigenvalues[step][0]
        gaps.append(gap)
    min_gap = min(gaps)
    min_step = gaps.index(min_gap)
    print(f"Minimum spectral gap : {min_gap:.6f}")
    print(f"At annealing step : {min_step}")

    # Test with a bigger instance
    separator("Bigger instance (5 qubits)")
    ising5 = create_ising_instance_random(5, weight_range=1.0, seed=42)
    all_eig5, _ = sim.simulate_evolution(ising5, nb_eigenvalues=5)
    best5, cost5 = exhaustive_search(ising5, maximize=False)
    print(f"Lowest eigenvalue at end : {all_eig5[-1][0]:.6f}")
    print(f"Exhaustive optimal : {cost5}")
    gaps5 = [all_eig5[s][1] - all_eig5[s][0] for s in range(len(all_eig5))]
    print(f"Min spectral gap : {min(gaps5):.6f} at step {gaps5.index(min(gaps5))}")

    separator("ALL TESTS COMPLETED")
    print("All Lab Session 3 TODOs executed successfully.")

if __name__ == '__main__':
    run_all_tests()
