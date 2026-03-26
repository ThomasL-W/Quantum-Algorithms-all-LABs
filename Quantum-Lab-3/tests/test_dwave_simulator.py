"""
Unit tests for the DWave simulator.
We check that the final eigenvalue matches the Ising optimal solution.
"""
import sys
import os
import unittest
import numpy as np
import networkx as nx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from Problem.IsingProblem import IsingProblem
from Algorithm.Classical_Algorithm.Exhaustive_search import exhaustive_search
from quantum_solver.dwave_simulator import DwaveSimulator

class TestDwaveSimulator(unittest.TestCase):

    def test_build_Hinit_shape(self):
        """H_init should be 2^n x 2^n."""
        sim = DwaveSimulator()
        H = sim.build_Hinit(3)
        self.assertEqual(H.shape, (8, 8))

    def test_build_Hfinal_shape(self):
        """H_final should be 2^n x 2^n."""
        G = nx.Graph()
        G.add_node(0, weight=0.5)
        G.add_node(1, weight=-0.3)
        G.add_edge(0, 1, weight=1.0)
        ising = IsingProblem(G)
        sim = DwaveSimulator()
        H = sim.build_Hfinal(ising)
        self.assertEqual(H.shape, (4, 4))

    def test_final_eigenvalue_matches_optimal(self):
        """The lowest eigenvalue at the last step should match the Ising optimal cost."""
        G = nx.Graph()
        G.add_node(0, weight=0.5)
        G.add_node(1, weight=-0.3)
        G.add_node(2, weight=0.1)
        G.add_edge(0, 1, weight=1.0)
        G.add_edge(1, 2, weight=-0.5)
        G.add_edge(0, 2, weight=0.8)
        ising = IsingProblem(G)
        # Get optimal with exhaustive search
        best_sol, best_cost = exhaustive_search(ising, maximize=False)
        # Simulate
        sim = DwaveSimulator()
        all_eigenvalues, all_eigenvectors = sim.simulate_evolution(ising, nb_eigenvalues=2)
        # The lowest eigenvalue at the last step should be close to the optimal cost
        final_lowest = all_eigenvalues[-1][0]
        self.assertAlmostEqual(final_lowest, best_cost, places=5,
                               msg=f"Final eigenvalue {final_lowest} != optimal cost {best_cost}")

    def test_annealing_schedule_length(self):
        """Schedule should have 101 points by default."""
        sim = DwaveSimulator()
        self.assertEqual(len(sim.annealing_schedule), 101)

    def test_annealing_schedule_endpoints(self):
        """A(0)=1, B(0)=0 at start and A(1)=0, B(1)=1 at end."""
        sim = DwaveSimulator()
        A_start, B_start = sim.annealing_schedule[0]
        A_end, B_end = sim.annealing_schedule[-1]
        self.assertAlmostEqual(A_start, 1.0)
        self.assertAlmostEqual(B_start, 0.0)
        self.assertAlmostEqual(A_end, 0.0)
        self.assertAlmostEqual(B_end, 1.0)

if __name__ == '__main__':
    unittest.main()
