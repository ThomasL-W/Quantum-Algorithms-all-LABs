import random
from itertools import product


class KnapsackProblem:
    """
    Represents a Knapsack optimization problem.

    Objective : maximize sum_i x_i * p_i
    Subject to: sum_i w_i * x_i < W
    where x_i in {0, 1}.
    """

    def __init__(self, prices, weights, capacity):
        """
        :param prices:   List of item prices/values p_i.
        :param weights:  List of item weights w_i.
        :param capacity: Maximum weight capacity W.
        :raises ValueError: If prices and weights have different lengths.
        """
        if len(prices) != len(weights):
            raise ValueError("prices and weights must have the same length.")
        self.prices   = list(prices)
        self.weights  = list(weights)
        self.capacity = capacity
        self.n        = len(prices)

    def is_feasible(self, solution) -> bool:
        """
        Checks whether a solution satisfies the weight constraint.

        :param solution: List of binary values {0, 1}.
        :return: True if sum_i w_i * x_i < W, False otherwise.
        """
        total_weight = sum(self.weights[i] * solution[i] for i in range(self.n))
        return total_weight < self.capacity

    def eval(self, solution) -> float:
        """
        Evaluates the total value of a solution.
        Returns 0 if the solution is infeasible.

        :param solution: List of binary values {0, 1}.
        :return: Total price as float, or 0 if infeasible.
        """
        if not self.is_feasible(solution):
            return 0.0
        return float(sum(self.prices[i] * solution[i] for i in range(self.n)))

    def generate_random_solution(self):
        """Generates a random {0, 1} binary solution."""
        return [random.randint(0, 1) for _ in range(self.n)]

    def gen_neighbor_sol(self, solution):
        """Generates a neighbor by flipping one random bit."""
        neighbor = list(solution)
        flip_idx = random.randint(0, self.n - 1)
        neighbor[flip_idx] = 1 - neighbor[flip_idx]
        return neighbor

    def generate_complete_search_space(self):
        """Generates all 2^n binary combinations for Knapsack."""
        return [list(combo) for combo in product([0, 1], repeat=self.n)]
