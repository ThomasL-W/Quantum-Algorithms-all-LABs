import numpy as np

def gradient_descent(func, grad_func, x0, y0, learning_rate=0.01, n_iterations=1000,
                     tolerance=1e-8, lr_schedule="constant"):
    """
    Gradient descent for a 2D function f(x, y).
    Parameters:
    -----------
    func : callable
        The function to minimize, takes (x, y) as arguments
    grad_func : callable
        The gradient function, returns (df/dx, df/dy)
    x0, y0 : float
        Initial coordinates
    learning_rate : float
        The step size (also called alpha or eta)
    n_iterations : int
        Maximum number of iterations
    tolerance : float
        Stop if the gradient norm is smaller than this
    lr_schedule : str
        Learning rate schedule: "constant", "decay" or "step"
    Returns:
    --------
    x_history, y_history, f_history : lists
        History of x, y and f(x,y) values at each iteration
    """
    # Initialize
    x = x0
    y = y0
    # Store the history for plotting later
    x_history = [x]
    y_history = [y]
    f_history = [func(x, y)]
    # Main loop
    for iteration in range(n_iterations):
        # Compute the gradient
        grad_x, grad_y = grad_func(x, y)
        # Check if the gradient is small enough to stop
        gradient_norm = np.sqrt(grad_x**2 + grad_y**2)
        if gradient_norm < tolerance:
            break
        # Compute the actual learning rate based on the schedule
        if lr_schedule == "constant":
            current_lr = learning_rate
        elif lr_schedule == "decay":
            current_lr = learning_rate / (1 + iteration * 0.01)
        elif lr_schedule == "step":
            how_many_halves = iteration // 100
            current_lr = learning_rate / (2 ** how_many_halves)
        else:
            current_lr = learning_rate
        # Update x and y
        x = x - current_lr * grad_x
        y = y - current_lr * grad_y
        # Store in history
        x_history.append(x)
        y_history.append(y)
        f_history.append(func(x, y))
    return x_history, y_history, f_history

def numerical_gradient(func, x, y, delta=1e-5):
    """
    Compute the gradient numerically using finite differences.
    This is useful when we don't have the analytical gradient.
    """
    # df/dx = (f(x+delta, y) - f(x-delta, y)) / (2*delta)
    grad_x = (func(x + delta, y) - func(x - delta, y)) / (2 * delta)
    # df/dy = (f(x, y+delta) - f(x, y-delta)) / (2*delta)
    grad_y = (func(x, y + delta) - func(x, y - delta)) / (2 * delta)
    return grad_x, grad_y
