import numpy as np


def qubo_to_ising(qubo_problem):
    """
    Converts a QuboProblem to an IsingProblem.

    Uses the substitution x_i = (1 + s_i) / 2, mapping x_i in {0,1} to s_i in {-1,+1}.

    :param qubo_problem: A QuboProblem instance.
    :return: An IsingProblem instance.
    """
    from .IsingProblem import IsingProblem

    Q = qubo_problem.matrix
    n = qubo_problem.n
    w = np.zeros(n)
    J = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                # Diagonal: Q[i,i]*x_i = Q[i,i]*(s_i+1)/2 -> linear contribution
                w[i] += Q[i, i] / 2
            elif j > i:
                # Upper triangular only: Q[i,j]*x_i*x_j = Q[i,j]/4*(s_i*s_j + s_i + s_j + 1)
                J[i, j] += Q[i, j] / 4
                w[i]    += Q[i, j] / 4
                w[j]    += Q[i, j] / 4

    return IsingProblem((w, J))


def ising_to_qubo(ising_problem):
    """
    Converts an IsingProblem to a QuboProblem.

    Uses the substitution s_i = 2*x_i - 1, mapping s_i in {-1,+1} to x_i in {0,1}.

    Expanding w_ij * s_i * s_j = w_ij * (2x_i-1)(2x_j-1)
                                = 4*w_ij*x_i*x_j - 2*w_ij*x_i - 2*w_ij*x_j + w_ij
    Expanding w_i * s_i        = w_i * (2*x_i - 1)
                                = 2*w_i*x_i - w_i

    :param ising_problem: An IsingProblem instance.
    :return: A QuboProblem instance with an upper triangular matrix.
    """
    from .QuboProblem import QuboProblem

    n = ising_problem.n
    Q = np.zeros((n, n))

    # Linear terms: w_i * s_i = 2*w_i*x_i - w_i  -> diagonal contribution
    for i in range(n):
        Q[i, i] += 2 * ising_problem.w[i]

    # Quadratic terms: w_ij*(2x_i-1)(2x_j-1) = 4*w_ij*x_i*x_j - 2*w_ij*x_i - 2*w_ij*x_j
    for (i, j), weight in ising_problem.J.items():
        Q[i, j] += 4 * weight    # quadratic term x_i*x_j (upper triangular)
        Q[i, i] -= 2 * weight    # linear term for x_i
        Q[j, j] -= 2 * weight    # linear term for x_j

    return QuboProblem(Q)
