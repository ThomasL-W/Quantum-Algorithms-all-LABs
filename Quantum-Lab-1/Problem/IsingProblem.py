import random
import numpy as np
import networkx as nx
from sympy import symbols, Integer
from itertools import product


class IsingProblem:
    """
    Represents an Ising optimization problem.

    Cost function: C = sum_i w_i * s_i + sum_{i<j} w_ij * s_i * s_j
    where s_i in {-1, +1}.
    """

    def __init__(self, input_data):
        """
        :param input_data: Either a networkx Graph (node/edge weight attributes)
                           or a tuple (w, J) where w is linear weights list
                           and J is a 2D array of quadratic weights.
        """
        if isinstance(input_data, nx.Graph):
            self._init_from_graph(input_data)
        elif isinstance(input_data, tuple) and len(input_data) == 2:
            self._init_from_weights(*input_data)
        else:
            raise ValueError("Input must be a networkx Graph or a tuple (w, J).")
        self._sympy_expr = None

    def _init_from_graph(self, graph):
        self.graph = graph
        self.n = graph.number_of_nodes()
        self.nodes = list(graph.nodes())
        self.w = {node: graph.nodes[node].get("weight", 0) for node in self.nodes}
        self.J = {
            (u, v): graph[u][v].get("weight", 1)
            for u, v in graph.edges()
        }

    def _init_from_weights(self, w, J):
        self.graph = None
        self.w = {i: w[i] for i in range(len(w))}
        self.n = len(w)
        self.nodes = list(range(self.n))
        J = np.array(J)
        self.J = {
            (i, j): J[i, j]
            for i in range(self.n)
            for j in range(i + 1, self.n)
            if J[i, j] != 0
        }

    def to_sympy_expr(self):
        """
        Converts the Ising problem to a sympy symbolic expression.
        Cached after the first computation.
        """
        if self._sympy_expr is not None:
            return self._sympy_expr

        expr = Integer(0)
        for node in self.nodes:
            if self.w[node] != 0:
                expr += self.w[node] * symbols(f"s_{node}")
        for (i, j), weight in self.J.items():
            expr += weight * symbols(f"s_{i}") * symbols(f"s_{j}")

        self._sympy_expr = expr
        return self._sympy_expr

    def get_cost(self, solution: dict) -> float:
        """
        Evaluates the Ising cost for a given solution.

        :param solution: Dict mapping variable name strings to {-1, +1}.
        :return: Numerical cost as a float.
        """
        expr = self.to_sympy_expr()
        subs_dict = {symbols(k): v for k, v in solution.items()}
        return float(expr.subs(subs_dict))

    def eval(self, solution: dict) -> float:
        """Alias for get_cost(), for compatibility with search functions."""
        return self.get_cost(solution)

    def generate_random_solution(self):
        """Generates a random {-1, +1} assignment for all spins."""
        return {f"s_{node}": random.choice([-1, 1]) for node in self.nodes}

    def gen_neighbor_sol(self, solution):
        """Generates a neighbor by flipping one random spin."""
        neighbor = dict(solution)
        flip_node = random.choice(self.nodes)
        key = f"s_{flip_node}"
        neighbor[key] = -neighbor[key]
        return neighbor

    def generate_complete_search_space(self):
        """Generates all {-1, +1}^n assignments for Ising."""
        return [
            {f"s_{self.nodes[i]}": combo[i] for i in range(self.n)}
            for combo in product([-1, 1], repeat=self.n)
        ]
