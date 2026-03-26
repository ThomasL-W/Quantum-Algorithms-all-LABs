import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh

# Pauli matrices
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
identity = np.eye(2, dtype=complex)

class DwaveSimulator:
    """
    Simulates the evolution of a D-Wave quantum annealer.
    The system evolves according to:
        H(s) = A(s) * H_init + B(s) * H_final
    where s goes from 0 to 1 over the annealing schedule.
    H_init is the transverse field Hamiltonian (sum of sigma_x on each qubit).
    H_final is the Ising Hamiltonian encoding the problem.
    """
    def __init__(self, n_steps=101):
        """
        n_steps : number of annealing steps (default 101 for s = 0, 0.01, ..., 1.0)
        """
        self.n_steps = n_steps
        # Linear annealing schedule with 101 points
        # A(s) goes from 1 to 0, B(s) goes from 0 to 1
        self.annealing_schedule = []
        for i in range(n_steps):
            s = i / (n_steps - 1)
            A_s = 1.0 - s
            B_s = s
            self.annealing_schedule.append((A_s, B_s))

    def _tensor_product_operator(self, operator, qubit_index, n):
        """
        Build the full 2^n x 2^n matrix for a single-qubit operator
        acting on qubit qubit_index, with identity on all other qubits.
        Uses the tensor product: I x I x ... x operator x ... x I
        """
        result = np.array([[1.0]], dtype=complex)
        for i in range(n):
            if i == qubit_index:
                result = np.kron(result, operator)
            else:
                result = np.kron(result, identity)
        return result

    def _tensor_product_two_operators(self, op1, qubit_i, op2, qubit_j, n):
        """
        Build the full 2^n x 2^n matrix for two single-qubit operators
        acting on qubits i and j respectively. This computes:
        I x ... x op1 x ... x op2 x ... x I
        """
        result = np.array([[1.0]], dtype=complex)
        for k in range(n):
            if k == qubit_i:
                result = np.kron(result, op1)
            elif k == qubit_j:
                result = np.kron(result, op2)
            else:
                result = np.kron(result, identity)
        return result

    def build_Hfinal(self, ising_problem):
        """
        Build the final Hamiltonian H_final from an Ising problem.
        H_final = sum_i w_i * sigma_z^(i) + sum_{i<j} J_ij * sigma_z^(i) * sigma_z^(j)
        where sigma_z^(i) means sigma_z acting on qubit i and identity elsewhere.
        """
        n = ising_problem.n
        dim = 2 ** n
        H_final = np.zeros((dim, dim), dtype=complex)
        # Linear terms: w_i * sigma_z on qubit i
        for node in ising_problem.nodes:
            w_i = ising_problem.w[node]
            if w_i != 0:
                H_final += w_i * self._tensor_product_operator(sigma_z, node, n)
        # Quadratic terms: J_ij * sigma_z^(i) * sigma_z^(j)
        for (i, j), weight in ising_problem.J.items():
            if weight != 0:
                H_final += weight * self._tensor_product_two_operators(sigma_z, i, sigma_z, j, n)
        return H_final

    def build_Hinit(self, n):
        """
        Build the initial Hamiltonian H_init.
        H_init = sum_i sigma_x^(i)
        This is the transverse field that puts the system in a superposition at the start.
        """
        dim = 2 ** n
        H_init = np.zeros((dim, dim), dtype=complex)
        for i in range(n):
            H_init += self._tensor_product_operator(sigma_x, i, n)
        return H_init

    def simulate_evolution(self, ising_problem, nb_eigenvalues=5):
        """
        Simulate the D-Wave annealing evolution.
        For each time step s in the annealing schedule, we build:
            H(s) = A(s) * H_init + B(s) * H_final
        and diagonalize it to get the eigenvalues.
        Returns a list of eigenvalue arrays, one per time step.
        """
        n = ising_problem.n
        H_init = self.build_Hinit(n)
        H_final = self.build_Hfinal(ising_problem)
        all_eigenvalues = []
        all_eigenvectors = []
        for step_idx in range(len(self.annealing_schedule)):
            A_s, B_s = self.annealing_schedule[step_idx]
            # Build H(s)
            H_s = A_s * H_init + B_s * H_final
            # Diagonalize - use eigh because H is hermitian
            # We get all eigenvalues sorted in ascending order
            eigenvalues, eigenvectors = eigh(H_s.real)
            # Keep only the nb_eigenvalues lowest
            all_eigenvalues.append(eigenvalues[:nb_eigenvalues])
            all_eigenvectors.append(eigenvectors[:, :nb_eigenvalues])
        return all_eigenvalues, all_eigenvectors

    def simulate_evolution_noisy(self, ising_problem, noise_std=0.05, nb_eigenvalues=5, noise_on_init=False):
        """
        Simulate the D-Wave evolution with noise on the Ising Hamiltonian.
        At each time step, we add noise to H_final:
            H_final_noisy = H_final + noise
        where noise is a random Hermitian matrix drawn from N(0, noise_std).
        If noise_on_init is True, we also add noise on H_init.
        """
        n = ising_problem.n
        dim = 2 ** n
        H_init = self.build_Hinit(n)
        H_final = self.build_Hfinal(ising_problem)
        all_eigenvalues = []
        for step_idx in range(len(self.annealing_schedule)):
            A_s, B_s = self.annealing_schedule[step_idx]
            # Add noise to H_final at each step
            noise_final = np.random.normal(0, noise_std, (dim, dim))
            noise_final = (noise_final + noise_final.T) / 2  # make it symmetric/hermitian
            H_final_noisy = H_final + noise_final
            # Optionally add noise to H_init too
            if noise_on_init:
                noise_init = np.random.normal(0, noise_std, (dim, dim))
                noise_init = (noise_init + noise_init.T) / 2
                H_init_noisy = H_init + noise_init
            else:
                H_init_noisy = H_init
            H_s = A_s * H_init_noisy + B_s * H_final_noisy
            eigenvalues, eigenvectors = eigh(H_s.real)
            all_eigenvalues.append(eigenvalues[:nb_eigenvalues])
        return all_eigenvalues

def plot_eigenvalues(all_eigenvalues, title="Eigenvalues during annealing"):
    """Plot the eigenvalues at each annealing step."""
    import matplotlib.pyplot as plt
    n_steps = len(all_eigenvalues)
    nb_eig = len(all_eigenvalues[0])
    for k in range(nb_eig):
        values = [all_eigenvalues[step][k] for step in range(n_steps)]
        plt.plot(range(n_steps), values, label=f"E_{k}")
    plt.xlabel("Annealing step")
    plt.ylabel("Energy")
    plt.title(title)
    plt.legend()

def plot_spectral_gap(all_eigenvalues, title="Spectral gap"):
    """Plot the spectral gap (E1 - E0) at each annealing step."""
    import matplotlib.pyplot as plt
    n_steps = len(all_eigenvalues)
    gaps = []
    for step in range(n_steps):
        gap = all_eigenvalues[step][1] - all_eigenvalues[step][0]
        gaps.append(gap)
    plt.plot(range(n_steps), gaps)
    plt.xlabel("Annealing step")
    plt.ylabel("Spectral gap (E1 - E0)")
    plt.title(title)
    min_gap = min(gaps)
    min_step = gaps.index(min_gap)
    plt.axhline(y=min_gap, color='r', linestyle='--', alpha=0.5, label=f"Min gap = {min_gap:.4f} at step {min_step}")
    plt.legend()

def create_ising_instance_random(n, weight_range=1.0, seed=None):
    """
    Create a random Ising problem instance with n variables.
    Weights are drawn uniformly from [-weight_range, +weight_range].
    """
    import networkx as nx
    if seed is not None:
        np.random.seed(seed)
    G = nx.Graph()
    for i in range(n):
        G.add_node(i, weight=np.random.uniform(-weight_range, weight_range))
    # Add edges between all pairs (fully connected)
    for i in range(n):
        for j in range(i + 1, n):
            G.add_edge(i, j, weight=np.random.uniform(-weight_range, weight_range))
    from Problem.IsingProblem import IsingProblem
    return IsingProblem(G)

def create_ising_with_duplicated_node(ising_problem, node_to_duplicate, coupling_strength=5.0):
    """
    Create a new Ising instance where one node is duplicated with a strong coupling.
    The duplicated node has the same weights as the original.
    A strong ferromagnetic coupling ensures both copies take the same value.
    """
    import networkx as nx
    from Problem.IsingProblem import IsingProblem
    n = ising_problem.n
    new_node = n  # the new duplicated node
    G = nx.Graph()
    # Copy original nodes
    for node in ising_problem.nodes:
        G.add_node(node, weight=ising_problem.w[node])
    # Add the duplicated node with same weight
    G.add_node(new_node, weight=ising_problem.w[node_to_duplicate])
    # Copy original edges
    for (i, j), w in ising_problem.J.items():
        G.add_edge(i, j, weight=w)
    # Add edges from duplicated node to all neighbors of the original
    for (i, j), w in ising_problem.J.items():
        if i == node_to_duplicate:
            G.add_edge(new_node, j, weight=w)
        elif j == node_to_duplicate:
            G.add_edge(new_node, i, weight=w)
    # Strong coupling between original and duplicate
    # Negative coupling = ferromagnetic = they want to be the same
    G.add_edge(node_to_duplicate, new_node, weight=-coupling_strength)
    return IsingProblem(G)
