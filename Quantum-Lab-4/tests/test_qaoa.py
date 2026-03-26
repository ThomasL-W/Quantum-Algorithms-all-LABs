"""
Unit tests for the QAOA components.
"""
import sys
import os
import unittest
import numpy as np
import networkx as nx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from Problem.IsingProblem import IsingProblem
from Problem.MaxCutProblem import MaxCutProblem
from quantum_solver.qaoa_solver.qaoa_mixers import add_ising_mixer_ham, add_ising_problem_ham, build_qaoa_circuit
from quantum_solver.qaoa_solver.qaoa_optimizer import QAOALocalOptimizer
from qiskit.circuit import QuantumCircuit

class TestQaoaMixers(unittest.TestCase):

    def setUp(self):
        G = nx.Graph()
        G.add_node(0, weight=0.0)
        G.add_node(1, weight=0.0)
        G.add_edge(0, 1, weight=-1.0)
        self.ising = IsingProblem(G)

    def test_add_mixer_returns_circuit_and_params(self):
        qc = QuantumCircuit(2)
        qc, params = add_ising_mixer_ham(qc, self.ising, 2)
        self.assertEqual(len(params), 1)
        self.assertEqual(params[0].name, 'beta')

    def test_add_problem_ham_returns_circuit_and_params(self):
        qc = QuantumCircuit(2)
        qc, params = add_ising_problem_ham(qc, self.ising, 2)
        self.assertEqual(len(params), 1)
        self.assertEqual(params[0].name, 'gamma')

    def test_build_qaoa_circuit_p1(self):
        qc, params = build_qaoa_circuit(self.ising, p=1)
        self.assertEqual(len(params), 2)  # gamma_0 and beta_0
        self.assertEqual(qc.num_qubits, 2)

    def test_build_qaoa_circuit_p3(self):
        qc, params = build_qaoa_circuit(self.ising, p=3)
        self.assertEqual(len(params), 6)  # 3 gammas + 3 betas

class TestQaoaOptimizer(unittest.TestCase):

    def test_run_without_optimization(self):
        G = nx.Graph()
        G.add_node(0, weight=0.0)
        G.add_node(1, weight=0.0)
        G.add_edge(0, 1, weight=-1.0)
        ising = IsingProblem(G)
        optimizer = QAOALocalOptimizer(p=1, shots=1024)
        exp_val, best_sol, angles = optimizer.run_without_optimization(ising)
        self.assertIsNotNone(best_sol)
        self.assertEqual(len(angles), 2)

    def test_optimize_small_instance(self):
        """Test that QAOA optimizer finds a reasonable solution on a 2-qubit problem."""
        G = nx.Graph()
        G.add_node(0, weight=0.0)
        G.add_node(1, weight=0.0)
        G.add_edge(0, 1, weight=-1.0)
        ising = IsingProblem(G)
        optimizer = QAOALocalOptimizer(p=2, shots=2048, opt_method='COBYLA')
        exp_val, best_sol, angles = optimizer.optimize(ising)
        # The optimal cost for this problem is -1 (spins aligned)
        self.assertLessEqual(exp_val, 0.0)

if __name__ == '__main__':
    unittest.main()
