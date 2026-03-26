"""
main.py — Test file for Lab Session 4.
Tests the QAOA implementation with Qiskit.
"""
import numpy as np
import networkx as nx
from Problem.IsingProblem import IsingProblem
from Problem.MaxCutProblem import MaxCutProblem
from Algorithm.Classical_Algorithm.Exhaustive_search import exhaustive_search
from quantum_solver.qaoa_solver.qaoa_mixers import build_qaoa_circuit
from quantum_solver.qaoa_solver.qaoa_optimizer import QAOALocalOptimizer

def separator(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

def run_all_tests():
    # TODO 2-4 : QAOA mixers and circuit
    separator("TODOs 2-4 | QAOA circuit building")
    # MaxCut instance from the lab: 5 nodes, edges to node 5
    G = nx.Graph()
    G.add_nodes_from([1, 2, 3, 4, 5])
    G.add_edges_from([(1, 5), (2, 5), (3, 5), (4, 5)])
    maxcut = MaxCutProblem(G)
    # Convert to Ising (MaxCut is already Ising-like)
    # Cost function: -s1*s5 - s2*s5 - s3*s5 - s4*s5
    G_ising = nx.Graph()
    for i in range(5):
        G_ising.add_node(i, weight=0.0)
    G_ising.add_edge(0, 4, weight=-1.0)  # s1*s5
    G_ising.add_edge(1, 4, weight=-1.0)  # s2*s5
    G_ising.add_edge(2, 4, weight=-1.0)  # s3*s5
    G_ising.add_edge(3, 4, weight=-1.0)  # s4*s5
    ising = IsingProblem(G_ising)
    print(f"Ising problem : {ising.n} qubits")
    best_sol, best_cost = exhaustive_search(ising, maximize=False)
    print(f"Optimal solution : {best_sol} cost={best_cost}")
    # Build QAOA circuit
    qc, params = build_qaoa_circuit(ising, p=1)
    print(f"QAOA circuit : {qc.num_qubits} qubits, {len(params)} parameters")
    print(f"Parameters : {[p.name for p in params]}")

    # TODO 5-8 : Run QAOA without optimization
    separator("TODOs 5-8 | Run QAOA (no optimization)")
    optimizer = QAOALocalOptimizer(p=1, shots=2048)
    exp_val, best_found, angles = optimizer.run_without_optimization(ising, angles=[0.5, 0.5])
    print(f"Angles : gamma={angles[0]:.4f}, beta={angles[1]:.4f}")
    print(f"Expectation value : {exp_val:.4f}")
    print(f"Best solution found : {best_found}")
    print(f"Best cost found : {optimizer.best_cost:.4f}")

    # TODO 9 : Optimize QAOA
    separator("TODO 9 | Optimize QAOA (p=1)")
    optimizer_opt = QAOALocalOptimizer(p=1, shots=2048, opt_method='COBYLA')
    exp_val_opt, best_opt, angles_opt = optimizer_opt.optimize(ising)
    print(f"Optimized expectation value : {exp_val_opt:.4f}")
    print(f"Best solution : {best_opt}")
    print(f"Best cost : {optimizer_opt.best_cost:.4f}")
    print(f"Optimal angles : {angles_opt}")
    print(f"Expected optimal cost : {best_cost}")

    # Try with deeper circuit
    separator("TODO 9 | Optimize QAOA (p=3)")
    optimizer_p3 = QAOALocalOptimizer(p=3, shots=2048, opt_method='COBYLA')
    exp_val_p3, best_p3, angles_p3 = optimizer_p3.optimize(ising)
    print(f"Optimized expectation value : {exp_val_p3:.4f}")
    print(f"Best solution : {best_p3}")
    print(f"Best cost : {optimizer_p3.best_cost:.4f}")

    separator("ALL TESTS COMPLETED")
    print("All Lab Session 4 TODOs executed successfully.")

if __name__ == '__main__':
    run_all_tests()
