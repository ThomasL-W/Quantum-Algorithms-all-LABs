import numpy as np
from scipy.linalg import issymmetric
from sympy import symbols, Integer
from itertools import product
import random


class QuboProblem:
    """
    Represents a QUBO (Quadratic Unconstrained Binary Optimization) problem.

    The cost function to minimize is: f(Q) = sum_i sum_j Q_ij * x_i * x_j
    where x_i in {0, 1} and Q is either upper triangular or symmetric.
    """

    def __init__(self, matrix):
        """
        :param matrix: A numpy/scipy matrix (upper triangular or symmetric).
        :raises ValueError: If the matrix is neither upper triangular nor symmetric.
        """
        m = np.array(matrix)
        is_upper_triangular = np.allclose(m, np.triu(m))
        is_symmetric = issymmetric(m, atol=1e-8)

        if not is_upper_triangular and not is_symmetric:
            raise ValueError("Matrix must be either upper triangular or symmetric.")

        self.matrix = m
        self.n = m.shape[0]
        self._sympy_expr = None

    def to_sympy_expr(self):
        """
        Converts the QUBO matrix to a sympy symbolic cost expression.
        Cached after the first computation.
        """
        if self._sympy_expr is not None:
            return self._sympy_expr

        expr = Integer(0)
        for i in range(self.n):
            for j in range(self.n):
                if self.matrix[i, j] != 0:
                    expr += self.matrix[i, j] * symbols(f"x_{i}") * symbols(f"x_{j}")

        self._sympy_expr = expr
        return self._sympy_expr

    def eval(self, solution) -> float:
        """
        Evaluates the cost via vector-matrix multiplication: x^T Q x.

        :param solution: List of binary values {0, 1} of length n.
        :return: Numerical cost as a float.
        """
        x = np.array(solution, dtype=float)
        return float(x @ self.matrix @ x)

    def gen_neighbor_sol(self, solution):
        """
        Generates a neighbor solution by flipping exactly one random bit.

        :param solution: List of binary values {0, 1}.
        :return: New list with one bit flipped.
        """
        neighbor = solution.copy()
        flip_idx = random.randint(0, self.n - 1)
        neighbor[flip_idx] = 1 - neighbor[flip_idx]
        return neighbor

    def generate_random_solution(self):
        """Generates a random {0, 1} binary solution."""
        return [random.randint(0, 1) for _ in range(self.n)]

    def generate_complete_search_space(self):
        """Generates all {0, 1}^n binary combinations for QUBO."""
        return [list(combo) for combo in product([0, 1], repeat=self.n)]
