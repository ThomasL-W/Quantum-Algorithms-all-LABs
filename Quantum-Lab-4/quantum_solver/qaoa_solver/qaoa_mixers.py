from qiskit.circuit import QuantumCircuit, Parameter
import numpy as np

def add_ising_mixer_ham(qc, ising_problem, n):
    """
    Add the mixing Hamiltonian part to the quantum circuit.
    The mixer is exp(-i * beta * sum_i sigma_x^i), which corresponds
    to applying Rx(2*beta) on each qubit.
    We use a parameterized gate so the angle beta can be set later.
    Returns the quantum circuit and the list of added parameters (betas).
    """
    beta = Parameter('beta')
    # Apply Rx rotation on each qubit
    for i in range(n):
        qc.rx(2 * beta, i)
    return qc, [beta]

def add_ising_problem_ham(qc, ising_problem, n):
    """
    Add the problem Hamiltonian part to the quantum circuit.
    For an Ising problem: H = sum_i w_i * sigma_z^i + sum_{i<j} J_ij * sigma_z^i * sigma_z^j
    The evolution exp(-i * gamma * H) is implemented as:
    - For linear terms w_i: Rz(2 * gamma * w_i) on qubit i
    - For quadratic terms J_ij: CNOT(i,j) - Rz(2 * gamma * J_ij) on j - CNOT(i,j)
    Returns the quantum circuit and the list of added parameters (gammas).
    """
    gamma = Parameter('gamma')
    # Linear terms: w_i * sigma_z^i -> Rz(2 * gamma * w_i) on qubit i
    for node in ising_problem.nodes:
        w_i = ising_problem.w[node]
        if w_i != 0:
            qc.rz(2 * gamma * w_i, node)
    # Quadratic terms: J_ij * sigma_z^i * sigma_z^j
    # Implemented with CNOT - Rz - CNOT pattern
    for (i, j), weight in ising_problem.J.items():
        if weight != 0:
            qc.cx(i, j)
            qc.rz(2 * gamma * weight, j)
            qc.cx(i, j)
    return qc, [gamma]

def build_qaoa_circuit(ising_problem, p=1):
    """
    Build a full QAOA circuit for an Ising problem with p layers.
    The circuit starts with Hadamard gates on all qubits (uniform superposition),
    then alternates p layers of problem Hamiltonian and mixing Hamiltonian.
    Each layer has its own gamma and beta parameters.
    Returns the quantum circuit and the list of all parameters.
    """
    n = ising_problem.n
    qc = QuantumCircuit(n, n)
    # Initial state: uniform superposition
    for i in range(n):
        qc.h(i)
    all_params = []
    for layer in range(p):
        # Problem Hamiltonian with gamma_layer
        gamma = Parameter(f'gamma_{layer}')
        # Linear terms
        for node in ising_problem.nodes:
            w_i = ising_problem.w[node]
            if w_i != 0:
                qc.rz(2 * gamma * w_i, node)
        # Quadratic terms
        for (i, j), weight in ising_problem.J.items():
            if weight != 0:
                qc.cx(i, j)
                qc.rz(2 * gamma * weight, j)
                qc.cx(i, j)
        all_params.append(gamma)
        # Mixing Hamiltonian with beta_layer
        beta = Parameter(f'beta_{layer}')
        for i in range(n):
            qc.rx(2 * beta, i)
        all_params.append(beta)
    # Measurement
    qc.measure(range(n), range(n))
    return qc, all_params
