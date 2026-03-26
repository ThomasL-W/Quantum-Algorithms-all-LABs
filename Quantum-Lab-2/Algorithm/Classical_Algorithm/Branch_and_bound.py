import copy

class BranchAndBound:
    """
    Branch and Bound algorithm for the Knapsack problem.
    The idea is to explore the solution space like a tree.
    At each node we decide to include or exclude an item.
    We use an upper bound to prune branches that can't lead to a better solution
    than the best one we've already found.
    The design is split into several methods:
    - solve(): main method that starts the algorithm
    - _branch_and_bound_recursive(): the recursive exploration
    - _compute_upper_bound(): computes the fractional upper bound
    - _is_feasible(): checks if current weight is under capacity
    """
    def __init__(self, knapsack_problem):
        """
        Initialize the B&B solver with a knapsack problem.
        Parameters:
        -----------
        knapsack_problem : KnapsackProblem or KnapsackVolumetric
            The knapsack problem instance to solve
        """
        self.problem = knapsack_problem
        self.prices = knapsack_problem.prices
        self.weights = knapsack_problem.weights
        self.capacity = knapsack_problem.capacity
        self.n = knapsack_problem.n
        # Check if the problem has volumes (KnapsackVolumetric)
        if hasattr(knapsack_problem, 'volumes') and hasattr(knapsack_problem, 'max_volume'):
            self.volumes = knapsack_problem.volumes
            self.max_volume = knapsack_problem.max_volume
            self.has_volume = True
        else:
            self.volumes = [0] * self.n
            self.max_volume = float('inf')
            self.has_volume = False
        # Sort items by value/weight ratio (descending) for better pruning
        self.sorted_indices = list(range(self.n))
        ratios = []
        for i in range(self.n):
            if self.weights[i] > 0:
                ratios.append(self.prices[i] / self.weights[i])
            else:
                ratios.append(float('inf'))
        # Sort by ratio descending
        self.sorted_indices.sort(key=lambda i: ratios[i], reverse=True)
        # Reorder the items according to the sorted order
        self.sorted_prices = [self.prices[i] for i in self.sorted_indices]
        self.sorted_weights = [self.weights[i] for i in self.sorted_indices]
        self.sorted_volumes = [self.volumes[i] for i in self.sorted_indices]
        # Best solution found so far
        self.best_value = 0
        self.best_solution = [0] * self.n

    def solve(self):
        """
        Solve the knapsack problem using branch and bound.
        Returns:
        --------
        best_solution : list
            Binary list indicating which items are selected (in original order)
        best_value : float
            The total value of the best solution
        """
        # Reset the best solution
        self.best_value = 0
        self.best_solution = [0] * self.n
        # Start the recursive exploration
        current_solution = [0] * self.n
        self._branch_and_bound_recursive(0, 0, 0, 0, current_solution)
        # Convert the solution back to original item order
        original_solution = [0] * self.n
        for i in range(self.n):
            original_index = self.sorted_indices[i]
            original_solution[original_index] = self.best_solution[i]
        return original_solution, self.best_value

    def _branch_and_bound_recursive(self, level, current_value, current_weight, current_volume, current_solution):
        """
        Recursive branch and bound exploration.
        At each level, we try two branches:
        1. Include the item at this level
        2. Exclude the item at this level
        We prune branches where the upper bound is less than the best known solution.
        """
        # Base case: we've considered all items
        if level == self.n:
            if current_value > self.best_value:
                self.best_value = current_value
                self.best_solution = list(current_solution)
            return
        # Check if the upper bound is worth exploring
        upper_bound = self._compute_upper_bound(level, current_value, current_weight, current_volume)
        if upper_bound <= self.best_value:
            # This branch can't do better than what we already have, prune it
            return
        # Branch 1: Include the current item (if feasible)
        new_weight = current_weight + self.sorted_weights[level]
        new_volume = current_volume + self.sorted_volumes[level]
        if new_weight < self.capacity and new_volume < self.max_volume:
            # It fits, so try including it
            current_solution[level] = 1
            new_value = current_value + self.sorted_prices[level]
            self._branch_and_bound_recursive(level + 1, new_value, new_weight, new_volume, current_solution)
        # Branch 2: Exclude the current item
        current_solution[level] = 0
        self._branch_and_bound_recursive(level + 1, current_value, current_weight, current_volume, current_solution)

    def _compute_upper_bound(self, level, current_value, current_weight, current_volume):
        """
        Compute the fractional upper bound using the greedy relaxation.
        We greedily add items (sorted by value/weight ratio) and allow fractional
        items to fill the remaining capacity.
        """
        remaining_capacity = self.capacity - current_weight
        remaining_volume = self.max_volume - current_volume
        bound = current_value
        for i in range(level, self.n):
            item_weight = self.sorted_weights[i]
            item_volume = self.sorted_volumes[i]
            item_price = self.sorted_prices[i]
            if item_weight <= remaining_capacity and item_volume <= remaining_volume:
                # Take the whole item
                bound += item_price
                remaining_capacity -= item_weight
                remaining_volume -= item_volume
            else:
                # Take a fraction of the item (limited by weight or volume)
                if item_weight > 0:
                    fraction_w = remaining_capacity / item_weight
                else:
                    fraction_w = 1.0
                if self.has_volume and item_volume > 0:
                    fraction_v = remaining_volume / item_volume
                else:
                    fraction_v = 1.0
                fraction = min(fraction_w, fraction_v)
                bound += item_price * fraction
                break  # after a fractional item, we stop
        return bound

    def _is_feasible(self, weight, volume=0):
        """Check if the current weight (and volume) is under capacity."""
        if weight > self.capacity:
            return False
        if self.has_volume and volume > self.max_volume:
            return False
        return True
