"""
Unit tests for the simulated annealing algorithm.
We test that SA converges to the known optimal solution for small QUBO instances.
"""
import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from Problem.QuboProblem import QuboProblem
from Algorithm.Classical_Algorithm.Simulated_annealing import simulated_annealing
from Algorithm.Classical_Algorithm.Exhaustive_search import exhaustive_search

class TestSimulatedAnnealing(unittest.TestCase):

    def test_small_qubo_symmetric(self):
        """Test SA on a small symmetric QUBO matrix."""
        Q = np.array([[ 2., -1.,  0.],
                      [-1.,  3., -1.],
                      [ 0., -1.,  2.]])
        qubo = QuboProblem(Q)
        best_exhaustive, cost_exhaustive = exhaustive_search(qubo, maximize=False)
        # Run SA multiple times to check it finds optimal at least once
        found_optimal = False
        for i in range(10):
            best_sa, cost_sa = simulated_annealing(qubo, initial_temperature=50.0,
                                                    decreasing_factor=0.95,
                                                    steps_per_temp=20,
                                                    temp_threshold=0.001)
            if cost_sa == cost_exhaustive:
                found_optimal = True
                break
        self.assertTrue(found_optimal, f"SA did not find optimal {best_exhaustive} cost {cost_exhaustive}")

    def test_small_qubo_upper_triangular(self):
        """Test SA on a small upper triangular QUBO matrix."""
        Q = np.array([[1., 2., 3.],
                      [0., 4., 5.],
                      [0., 0., 6.]])
        qubo = QuboProblem(Q)
        best_exhaustive, cost_exhaustive = exhaustive_search(qubo, maximize=False)
        found_optimal = False
        for i in range(10):
            best_sa, cost_sa = simulated_annealing(qubo, initial_temperature=100.0,
                                                    decreasing_factor=0.9,
                                                    steps_per_temp=15,
                                                    temp_threshold=0.01)
            if cost_sa == cost_exhaustive:
                found_optimal = True
                break
        self.assertTrue(found_optimal)

    def test_different_cooling_schedules(self):
        """Test that all three cooling schedules work and converge."""
        Q = np.array([[ 2., -1.],
                      [-1.,  3.]])
        qubo = QuboProblem(Q)
        best_exhaustive, cost_exhaustive = exhaustive_search(qubo, maximize=False)
        for schedule in ["exponential", "linear", "geometric"]:
            found = False
            for attempt in range(15):
                best, cost = simulated_annealing(qubo, initial_temperature=50.0,
                                                  decreasing_factor=0.9,
                                                  steps_per_temp=10,
                                                  temp_threshold=0.01,
                                                  cooling_schedule=schedule)
                if cost == cost_exhaustive:
                    found = True
                    break
            self.assertTrue(found, f"Schedule '{schedule}' did not converge")

    def test_sa_returns_binary_solution(self):
        """Check that SA returns a valid binary vector."""
        Q = np.array([[1., 0.], [0., 1.]])
        qubo = QuboProblem(Q)
        best, cost = simulated_annealing(qubo, initial_temperature=10.0,
                                          decreasing_factor=0.9,
                                          steps_per_temp=5,
                                          temp_threshold=0.1)
        for val in best:
            self.assertIn(val, [0, 1])
        self.assertEqual(len(best), qubo.n)

if __name__ == '__main__':
    unittest.main()
