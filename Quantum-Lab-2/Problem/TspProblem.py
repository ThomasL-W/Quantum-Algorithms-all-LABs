import random
import numpy as np
from itertools import permutations


class TspProblem:
    """
    Represents a Travelling Salesman Problem (TSP).

    Find the shortest route visiting each city exactly once
    and returning to the origin city.
    """

    def __init__(self, distance_matrix, city_names=None):
        """
        :param distance_matrix: 2D array where [i][j] is distance from city i to j.
        :param city_names:      Optional list of city name strings.
        """
        self.distance_matrix = np.array(distance_matrix)
        self.n = self.distance_matrix.shape[0]
        self.city_names = city_names if city_names else [f"City_{i}" for i in range(self.n)]

    def eval(self, solution) -> float:
        """
        Evaluates the total distance of a TSP route (including return to start).

        :param solution: List of city indices representing the visit order.
        :return: Total round-trip distance as a float.
        """
        total = 0.0
        for i in range(self.n):
            total += self.distance_matrix[solution[i]][solution[(i + 1) % self.n]]
        return total

    def generate_complete_search_space(self):
        """
        Generates all permutations of cities, fixing city 0 as the start
        to avoid counting rotations as different tours.
        """
        other_cities = list(range(1, self.n))
        return [[0] + list(perm) for perm in permutations(other_cities)]

    def generate_random_solution(self):
        """Generates a random permutation of cities, fixing city 0 as the start."""
        other_cities = list(range(1, self.n))
        random.shuffle(other_cities)
        return [0] + other_cities

    def gen_neighbor_sol(self, solution):
        """Generates a neighbor by swapping two random cities (not the starting city 0)."""
        neighbor = list(solution)
        i, j = random.sample(range(1, self.n), 2)
        neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        return neighbor

    def display_solution(self, solution, pos=None):
        """
        Displays the TSP route on a 2D plot.

        :param solution: List of city indices.
        :param pos:      Optional dict {city_index: (x, y)}. Defaults to circular layout.
        """
        import matplotlib.pyplot as plt

        if pos is None:
            angles = np.linspace(0, 2 * np.pi, self.n, endpoint=False)
            pos = {i: (np.cos(a), np.sin(a)) for i, a in enumerate(angles)}

        plt.figure(figsize=(7, 6))
        for i in range(self.n):
            from_city, to_city = solution[i], solution[(i + 1) % self.n]
            plt.plot([pos[from_city][0], pos[to_city][0]],
                     [pos[from_city][1], pos[to_city][1]], "b-", alpha=0.6)
        for city, (x, y) in pos.items():
            plt.scatter(x, y, s=300, c="black", zorder=5)
            plt.text(x, y + 0.08, self.city_names[city], ha="center", fontsize=10, color="darkred")

        total_dist = self.eval(solution)
        route_str  = " -> ".join(self.city_names[c] for c in solution) + f" -> {self.city_names[solution[0]]}"
        plt.title(f"TSP Solution (distance={total_dist:.2f})\n{route_str}", fontsize=9)
        plt.axis("off")
        plt.tight_layout()
        plt.show()
