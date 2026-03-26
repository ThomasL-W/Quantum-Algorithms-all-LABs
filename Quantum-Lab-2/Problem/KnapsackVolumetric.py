import random
from itertools import product

class KnapsackVolumetric:
    """
    Represents a Knapsack problem with both weight and volume constraints.
    Objective : maximize sum_i x_i * p_i
    Subject to: sum_i w_i * x_i < W   (weight constraint)
                sum_i v_i * x_i < V   (volume constraint)
    where x_i in {0, 1}.
    The normal KnapsackProblem is just a special case where all volumes are 0
    and max_volume is infinity.
    """
    def __init__(self, prices, weights, capacity, volumes=None, max_volume=None):
        if len(prices) != len(weights):
            raise ValueError("prices and weights must have the same length.")
        self.prices = list(prices)
        self.weights = list(weights)
        self.capacity = capacity
        self.n = len(prices)
        # If no volumes given, set them all to 0 (behaves like regular Knapsack)
        if volumes is None:
            self.volumes = [0] * self.n
        else:
            if len(volumes) != self.n:
                raise ValueError("volumes must have the same length as prices.")
            self.volumes = list(volumes)
        # If no max volume given, set to infinity
        if max_volume is None:
            self.max_volume = float('inf')
        else:
            self.max_volume = max_volume

    def is_feasible(self, solution) -> bool:
        """Checks if a solution satisfies both weight and volume constraints."""
        # Check weight constraint
        total_weight = 0
        for i in range(self.n):
            total_weight += self.weights[i] * solution[i]
        if total_weight >= self.capacity:
            return False
        # Check volume constraint
        total_volume = 0
        for i in range(self.n):
            total_volume += self.volumes[i] * solution[i]
        if total_volume >= self.max_volume:
            return False
        return True

    def eval(self, solution) -> float:
        """Evaluates the total value. Returns 0 if infeasible."""
        if not self.is_feasible(solution):
            return 0.0
        total_value = 0.0
        for i in range(self.n):
            total_value += self.prices[i] * solution[i]
        return total_value

    def generate_random_solution(self):
        """Generate a random binary solution."""
        solution = []
        for i in range(self.n):
            solution.append(random.randint(0, 1))
        return solution

    def gen_neighbor_sol(self, solution):
        """Generate a neighbor by flipping one random bit."""
        neighbor = list(solution)
        idx = random.randint(0, self.n - 1)
        neighbor[idx] = 1 - neighbor[idx]
        return neighbor

    def generate_complete_search_space(self):
        """Generate all 2^n binary combinations."""
        all_solutions = []
        for combo in product([0, 1], repeat=self.n):
            all_solutions.append(list(combo))
        return all_solutions
