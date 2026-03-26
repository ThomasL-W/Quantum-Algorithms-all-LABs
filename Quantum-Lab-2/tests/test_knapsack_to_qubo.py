"""
Unit tests for the knapsack to QUBO converter.
We check that the optimal solution is preserved after converting.
"""
import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from Problem.KnapsackProblem import KnapsackProblem
from Problem.Converter import knapsack_to_qubo
from Algorithm.Classical_Algorithm.Exhaustive_search import exhaustive_search

class TestKnapsackToQubo(unittest.TestCase):

    def test_optimal_preserved(self):
        """The optimal knapsack solution should also be optimal in QUBO form."""
        knapsack = KnapsackProblem(prices=[10, 6, 5, 4, 3],
                                    weights=[5, 4, 3, 2, 1],
                                    capacity=8)
        best_knapsack, val_knapsack = exhaustive_search(knapsack, maximize=True)
        qubo = knapsack_to_qubo(knapsack)
        best_qubo, cost_qubo = exhaustive_search(qubo, maximize=False)
        self.assertEqual(best_qubo, best_knapsack,
                         f"QUBO optimal {best_qubo} != Knapsack optimal {best_knapsack}")

    def test_small_instance(self):
        """Test on a very small 2-item instance."""
        knapsack = KnapsackProblem(prices=[5, 3], weights=[4, 2], capacity=5)
        best_knapsack, val_knapsack = exhaustive_search(knapsack, maximize=True)
        qubo = knapsack_to_qubo(knapsack)
        best_qubo, cost_qubo = exhaustive_search(qubo, maximize=False)
        self.assertEqual(val_knapsack, 5.0)

if __name__ == '__main__':
    unittest.main()
