import numpy as np
from scipy.optimize import minimize
from qiskit_aer import AerSimulator
from .qaoa_mixers import build_qaoa_circuit

class QAOALocalOptimizer:
    """
    QAOA optimizer that builds and optimizes QAOA circuits for Ising problems.
    Uses scipy.optimize.minimize to find the best angles gamma and beta.
    """
    def __init__(self, simulator=None, gamma_bounds=(0, np.pi), beta_bounds=(0, np.pi),
                 p=1, shots=2048, opt_method='COBYLA'):
        """
        Parameters:
        simulator : AerSimulator instance (if None, creates a default noiseless one)
        gamma_bounds : tuple (min, max) for gamma angles
        beta_bounds : tuple (min, max) for beta angles
        p : number of QAOA layers
        shots : number of shots per circuit execution
        opt_method : scipy optimization method (COBYLA, Nelder-Mead, Powell, etc.)
        """
        if simulator is None:
            self.simulator = AerSimulator()
        else:
            self.simulator = simulator
        self.gamma_bounds = gamma_bounds
        self.beta_bounds = beta_bounds
        self.p = p
        self.shots = shots
        self.opt_method = opt_method
        # Keep track of best solution found during optimization
        self.best_solution = None
        self.best_cost = float('inf')

    def _run_circuit(self, angles, qc):
        """
        Run the quantum circuit with the given angles.
        Binds the parameter values to the circuit and executes it on the simulator.
        Returns the dictionary of measurement counts.
        """
        # The angles list has 2*p entries: gamma_0, beta_0, gamma_1, beta_1, ...
        # Map them to the circuit parameters
        params = qc.parameters
        param_dict = {}
        for i, param in enumerate(sorted(params, key=lambda p: p.name)):
            param_dict[param] = angles[i]
        bound_qc = qc.assign_parameters(param_dict)
        result = self.simulator.run(bound_qc, shots=self.shots).result()
        counts = result.get_counts()
        return counts

    def get_expectation_value(self, angles, qc, problem):
        """
        Run the circuit and compute the expectation value of the Ising cost function.
        The expectation value is the average cost over all shots.
        We convert the qubit measurements (0/1) to Ising spins (-1/+1) before
        evaluating the cost function.
        Qiskit uses little-endian convention: the first qubit is the rightmost bit.
        """
        counts = self._run_circuit(angles, qc)
        total_cost = 0.0
        total_shots = 0
        for bitstring, count in counts.items():
            # Convert bitstring to spin assignment
            # Qiskit is little-endian: bitstring[0] is the last qubit
            # We reverse it to get qubit 0 first
            bits = bitstring[::-1]
            solution = {}
            for i, bit in enumerate(bits):
                # 0 -> +1, 1 -> -1
                if bit == '0':
                    solution[f"s_{problem.nodes[i]}"] = 1
                else:
                    solution[f"s_{problem.nodes[i]}"] = -1
            cost = problem.eval(solution)
            total_cost += cost * count
            total_shots += count
            # Track best solution
            if cost < self.best_cost:
                self.best_cost = cost
                self.best_solution = solution
        expectation = total_cost / total_shots
        return expectation

    def run_without_optimization(self, problem, p=None, angles=None):
        """
        Build and run the QAOA circuit with given angles (no optimization).
        Returns the expectation value, best solution, and angles used.
        """
        if p is None:
            p = self.p
        qc, params = build_qaoa_circuit(problem, p=p)
        if angles is None:
            # Random angles
            angles = []
            for i in range(p):
                angles.append(np.random.uniform(self.gamma_bounds[0], self.gamma_bounds[1]))
                angles.append(np.random.uniform(self.beta_bounds[0], self.beta_bounds[1]))
        self.best_solution = None
        self.best_cost = float('inf')
        exp_val = self.get_expectation_value(angles, qc, problem)
        return exp_val, self.best_solution, angles

    def optimize(self, problem, p=None):
        """
        Optimize the QAOA angles using scipy.optimize.minimize.
        Returns the expectation value, best solution, and optimized angles.
        """
        if p is None:
            p = self.p
        qc, params = build_qaoa_circuit(problem, p=p)
        self.best_solution = None
        self.best_cost = float('inf')
        # Objective function for the optimizer
        def objective(angles):
            return self.get_expectation_value(angles, qc, problem)
        # Initial random angles
        x0 = []
        bounds = []
        for i in range(p):
            x0.append(np.random.uniform(self.gamma_bounds[0], self.gamma_bounds[1]))
            bounds.append(self.gamma_bounds)
            x0.append(np.random.uniform(self.beta_bounds[0], self.beta_bounds[1]))
            bounds.append(self.beta_bounds)
        x0 = np.array(x0)
        # Run the optimization
        if self.opt_method in ['COBYLA', 'Nelder-Mead', 'Powell']:
            # These methods don't support bounds directly, so we don't pass them
            result = minimize(objective, x0, method=self.opt_method,
                              options={'maxiter': 200})
        else:
            result = minimize(objective, x0, method=self.opt_method, bounds=bounds,
                              options={'maxiter': 200})
        final_exp_val = result.fun
        optimized_angles = result.x
        return final_exp_val, self.best_solution, optimized_angles
