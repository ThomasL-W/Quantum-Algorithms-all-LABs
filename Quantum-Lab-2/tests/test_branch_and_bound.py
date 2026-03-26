"""
Unit tests for the Branch and Bound algorithm.
We verify B&B finds the same optimal solution as exhaustive search.
"""
import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from Problem.KnapsackProblem import KnapsackProblem
from Problem.KnapsackVolumetric import KnapsackVolumetric
from Algorithm.Classical_Algorithm.Branch_and_bound import BranchAndBound
from Algorithm.Classical_Algorithm.Exhaustive_search import exhaustive_search

class TestBranchAndBound(unittest.TestCase):

    def test_small_knapsack(self):
        """Test B&B on a small knapsack instance."""
        knapsack = KnapsackProblem(prices=[10, 6, 5, 4, 3],
                                    weights=[5, 4, 3, 2, 1],
                                    capacity=8)
        best_exhaustive, val_exhaustive = exhaustive_search(knapsack, maximize=True)
        bb = BranchAndBound(knapsack)
        best_bb, val_bb = bb.solve()
        self.assertEqual(val_bb, val_exhaustive,
                         f"B&B value {val_bb} != exhaustive value {val_exhaustive}")

    def test_trivial_knapsack(self):
        """Everything fits in the bag."""
        knapsack = KnapsackProblem(prices=[1, 2, 3], weights=[1, 1, 1], capacity=10)
        bb = BranchAndBound(knapsack)
        best_bb, val_bb = bb.solve()
        self.assertEqual(val_bb, 6.0)

    def test_nothing_fits(self):
        """No item fits in the bag."""
        knapsack = KnapsackProblem(prices=[10, 20, 30], weights=[100, 200, 300], capacity=5)
        bb = BranchAndBound(knapsack)
        best_bb, val_bb = bb.solve()
        self.assertEqual(val_bb, 0)

    def test_knapsack_volumetric(self):
        """Test B&B on a volumetric knapsack."""
        knapsack_vol = KnapsackVolumetric(prices=[10, 6, 5, 4],
                                           weights=[5, 4, 3, 2],
                                           capacity=8,
                                           volumes=[3, 2, 4, 1],
                                           max_volume=6)
        best_exhaustive, val_exhaustive = exhaustive_search(knapsack_vol, maximize=True)
        bb = BranchAndBound(knapsack_vol)
        best_bb, val_bb = bb.solve()
        self.assertEqual(val_bb, val_exhaustive)

    def test_volumetric_zero_volumes_equals_regular(self):
        """KnapsackVolumetric with zero volumes should give same result as regular."""
        knapsack_reg = KnapsackProblem(prices=[10, 6, 5], weights=[5, 4, 3], capacity=8)
        knapsack_vol = KnapsackVolumetric(prices=[10, 6, 5], weights=[5, 4, 3], capacity=8,
                                           volumes=[0, 0, 0], max_volume=float('inf'))
        bb_reg = BranchAndBound(knapsack_reg)
        bb_vol = BranchAndBound(knapsack_vol)
        _, val_reg = bb_reg.solve()
        _, val_vol = bb_vol.solve()
        self.assertEqual(val_reg, val_vol)

if __name__ == '__main__':
    unittest.main()
