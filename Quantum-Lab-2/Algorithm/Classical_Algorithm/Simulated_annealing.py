import numpy as np
import random
import math

def simulated_annealing(qubo_problem, initial_temperature=100.0, decreasing_factor=0.95,
                        steps_per_temp=10, temp_threshold=0.01, cooling_schedule="exponential"):
    """
    Simulated annealing algorithm to minimize a QUBO problem.
    The idea is to start with a high temperature and then gradually decrease it.
    At high temperatures we accept worse solutions more often, and as the temperature
    decreases we become more and more strict about accepting worse solutions.

    Parameters:
    -----------
    qubo_problem : QuboProblem
        The QUBO problem to solve (we use the matrix Q to evaluate xTQx)
    initial_temperature : float
        The starting temperature T0
    decreasing_factor : float
        The alpha constant for the cooling schedule (between 0 and 1)
    steps_per_temp : int
        How many iterations we do at each temperature level before cooling
    temp_threshold : float
        When the temperature goes below this we stop the algorithm
    cooling_schedule : str
        The type of cooling schedule: "exponential", "linear" or "geometric"

    Returns:
    --------
    best_solution : list
        The best solution found (list of 0s and 1s)
    best_cost : float
        The cost of the best solution
    """
    n = qubo_problem.n
    Q = qubo_problem.matrix
    # Generate a random initial solution
    current_solution = [random.randint(0, 1) for i in range(n)]
    # Calculate initial cost using xTQx
    # TODO 3: we use delta evaluation instead of full xTQx each time
    # The idea is that when we flip bit k, the change in cost is:
    # delta = (1 - 2*x_k) * (Q[k,k] + sum_j!=k Q[k,j]*x_j + sum_j!=k Q[j,k]*x_j)
    # This is O(n) instead of O(n^2) for full matrix multiplication
    current_x = np.array(current_solution, dtype=float)
    current_cost = float(current_x @ Q @ current_x)
    # Keep track of the best solution we've found
    best_solution = list(current_solution)
    best_cost = current_cost
    # Set up the temperature
    temperature = initial_temperature
    # Calculate initial temperature for linear schedule
    if cooling_schedule == "linear":
        # For linear we decrease by a fixed amount each time
        linear_decrease = initial_temperature * (1 - decreasing_factor)
        if linear_decrease <= 0:
            linear_decrease = 0.1  # just in case
    # Main loop - keep going until temperature is below threshold
    while temperature > temp_threshold:
        # Do several iterations at this temperature
        for step in range(steps_per_temp):
            # Pick a random bit to flip
            flip_index = random.randint(0, n - 1)
            # TODO 3: Compute the delta cost efficiently
            # Instead of recomputing the full xTQx, we compute only the change
            # when flipping bit at flip_index
            x_k = current_solution[flip_index]
            # Calculate delta = change in cost if we flip bit k
            # delta = (1 - 2*x_k) * (Q[k][k] + sum over j of (Q[k][j] + Q[j][k]) * x_j for j != k)
            delta = 0.0
            delta += Q[flip_index][flip_index]  # diagonal term
            for j in range(n):
                if j != flip_index:
                    delta += (Q[flip_index][j] + Q[j][flip_index]) * current_solution[j]
            delta = (1 - 2 * x_k) * delta
            # Decide whether to accept this move
            if delta < 0:
                # The new solution is better (lower cost), always accept
                accept = True
            else:
                # The new solution is worse, accept with probability exp(-delta/T)
                probability = math.exp(-delta / temperature)
                random_number = random.random()
                if random_number < probability:
                    accept = True
                else:
                    accept = False
            # If we accept, flip the bit and update the cost
            if accept:
                current_solution[flip_index] = 1 - current_solution[flip_index]
                current_cost = current_cost + delta
                # Check if this is the best solution we've seen
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_solution = list(current_solution)  # make a copy
        # Cool down the temperature based on the schedule
        if cooling_schedule == "exponential":
            temperature = temperature * decreasing_factor
        elif cooling_schedule == "linear":
            temperature = temperature - linear_decrease
            if temperature < 0:
                temperature = 0  # don't go negative
        elif cooling_schedule == "geometric":
            # geometric is basically the same as exponential in practice
            temperature = temperature * decreasing_factor
        else:
            # default to exponential if unknown schedule
            temperature = temperature * decreasing_factor
    return best_solution, best_cost

def simulated_annealing_tsp(tsp_problem, initial_temperature=1000.0, decreasing_factor=0.99,
                            steps_per_temp=20, temp_threshold=0.1, cooling_schedule="exponential"):
    """
    Simulated annealing adapted for the TSP problem.
    For TSP we can't flip bits, instead we swap two cities in the route.
    The solution is a permutation of cities (with city 0 fixed at the start).

    Parameters:
    -----------
    tsp_problem : TspProblem
        The TSP problem to solve
    initial_temperature : float
        Starting temperature
    decreasing_factor : float
        Cooling factor alpha
    steps_per_temp : int
        Number of iterations per temperature level
    temp_threshold : float
        Minimum temperature threshold
    cooling_schedule : str
        Type of cooling: "exponential", "linear" or "geometric"

    Returns:
    --------
    best_solution : list
        Best route found
    best_cost : float
        Distance of the best route
    """
    n = tsp_problem.n
    distance_matrix = tsp_problem.distance_matrix
    # Generate random initial solution (fix city 0 at start)
    other_cities = list(range(1, n))
    random.shuffle(other_cities)
    current_solution = [0] + other_cities
    # Calculate initial cost
    current_cost = 0.0
    for i in range(n):
        from_city = current_solution[i]
        to_city = current_solution[(i + 1) % n]
        current_cost += distance_matrix[from_city][to_city]
    # Best solution tracking
    best_solution = list(current_solution)
    best_cost = current_cost
    temperature = initial_temperature
    # For linear cooling
    if cooling_schedule == "linear":
        linear_decrease = initial_temperature * (1 - decreasing_factor)
        if linear_decrease <= 0:
            linear_decrease = 0.5
    while temperature > temp_threshold:
        for step in range(steps_per_temp):
            # Generate neighbor: swap two random cities (not city 0)
            idx_i = random.randint(1, n - 1)
            idx_j = random.randint(1, n - 1)
            while idx_j == idx_i:
                idx_j = random.randint(1, n - 1)
            # Create the neighbor solution
            neighbor = list(current_solution)
            neighbor[idx_i], neighbor[idx_j] = neighbor[idx_j], neighbor[idx_i]
            # Calculate neighbor cost (full recalculation for TSP)
            neighbor_cost = 0.0
            for i in range(n):
                from_city = neighbor[i]
                to_city = neighbor[(i + 1) % n]
                neighbor_cost += distance_matrix[from_city][to_city]
            # Calculate delta
            delta = neighbor_cost - current_cost
            # Acceptance criterion
            if delta < 0:
                accept = True
            else:
                prob = math.exp(-delta / temperature)
                if random.random() < prob:
                    accept = True
                else:
                    accept = False
            if accept:
                current_solution = neighbor
                current_cost = neighbor_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_solution = list(current_solution)
        # Cool down
        if cooling_schedule == "exponential":
            temperature = temperature * decreasing_factor
        elif cooling_schedule == "linear":
            temperature = temperature - linear_decrease
            if temperature < 0:
                temperature = 0
        elif cooling_schedule == "geometric":
            temperature = temperature * decreasing_factor
        else:
            temperature = temperature * decreasing_factor
    return best_solution, best_cost
