import numpy as np

def qubo_to_ising(qubo_problem):
    """
    Converts a QuboProblem to an IsingProblem.
    Uses the substitution x_i = (1 + s_i) / 2, mapping x_i in {0,1} to s_i in {-1,+1}.
    """
    from .IsingProblem import IsingProblem
    Q = qubo_problem.matrix
    n = qubo_problem.n
    w = np.zeros(n)
    J = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                w[i] += Q[i, i] / 2
            elif j > i:
                J[i, j] += Q[i, j] / 4
                w[i]    += Q[i, j] / 4
                w[j]    += Q[i, j] / 4
    return IsingProblem((w, J))

def ising_to_qubo(ising_problem):
    """
    Converts an IsingProblem to a QuboProblem.
    Uses the substitution s_i = 2*x_i - 1, mapping s_i in {-1,+1} to x_i in {0,1}.
    Expanding w_ij * s_i * s_j = 4*w_ij*x_i*x_j - 2*w_ij*x_i - 2*w_ij*x_j + w_ij
    Expanding w_i * s_i = 2*w_i*x_i - w_i
    """
    from .QuboProblem import QuboProblem
    n = ising_problem.n
    Q = np.zeros((n, n))
    # Linear terms
    for i in range(n):
        Q[i, i] += 2 * ising_problem.w[i]
    # Quadratic terms
    for (i, j), weight in ising_problem.J.items():
        Q[i, j] += 4 * weight
        Q[i, i] -= 2 * weight
        Q[j, j] -= 2 * weight
    return QuboProblem(Q)

def knapsack_to_qubo(knapsack_problem, penalty=None):
    """
    Converts a Knapsack problem to a QUBO problem using penalty functions.
    The knapsack problem is:
        maximize sum_i p_i * x_i
        subject to: sum_i w_i * x_i < W
    We transform it to QUBO minimization:
        minimize -sum_i p_i * x_i + P * (sum_i w_i * x_i - W_eff)^2
    where W_eff = W-1 because the constraint is strict (< not <=).
    The QUBO matrix Q is:
    Q[i][i] = -p_i + P*w_i^2 - 2*P*W_eff*w_i   (diagonal)
    Q[i][j] = 2*P*w_i*w_j for i < j              (upper triangular)
    The constant P*W_eff^2 doesn't change the optimal so we ignore it.
    """
    from .QuboProblem import QuboProblem
    n = knapsack_problem.n
    prices = knapsack_problem.prices
    weights = knapsack_problem.weights
    W = knapsack_problem.capacity
    # Set penalty if not given
    if penalty is None:
        penalty = sum(prices) + 1
    P = penalty
    # Use W-1 because constraint is strict (< W)
    W_eff = W - 1
    # Build QUBO matrix
    Q = np.zeros((n, n))
    # Diagonal terms
    for i in range(n):
        Q[i][i] = -prices[i] + P * weights[i] * weights[i] - 2 * P * W_eff * weights[i]
    # Off-diagonal (upper triangular)
    for i in range(n):
        for j in range(i + 1, n):
            Q[i][j] = 2 * P * weights[i] * weights[j]
    return QuboProblem(Q)
