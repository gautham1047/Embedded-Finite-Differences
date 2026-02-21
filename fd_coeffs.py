import numpy as np
from math import factorial

# Calculate finite difference coefficients for a given stencil and derivative order.
def calculate_fd_coefficients(stencil, derivative_order):
    stencil = np.array(stencil)
    N = len(stencil)
    
    if derivative_order >= N:
        raise ValueError(f"Derivative order {derivative_order} must be less than stencil length {N}")
    
    # Build Vandermonde matrix
    M = np.vander(stencil, N, increasing=True).T
    
    # Build RHS vector: zeros except at position derivative_order
    rhs = np.zeros(N)
    rhs[derivative_order] = factorial(derivative_order)
    
    # Solve the system
    coefficients = np.linalg.solve(M, rhs)
    
    return coefficients, derivative_order


def forward_difference(derivative_order, accuracy_order=None):
    if accuracy_order is None:
        accuracy_order = 1
    
    # Number of points needed
    num_points = derivative_order + accuracy_order
    
    # Forward stencil: [0, 1, 2, ..., num_points-1]
    stencil = np.arange(num_points)
    
    return calculate_fd_coefficients(stencil, derivative_order)


def backward_difference(derivative_order, accuracy_order=None):
    if accuracy_order is None:
        accuracy_order = 1
    
    # Number of points needed
    num_points = derivative_order + accuracy_order
    
    # Backward stencil: [0, -1, -2, ..., -(num_points-1)]
    stencil = -np.arange(num_points)[::-1]
    
    return calculate_fd_coefficients(stencil, derivative_order)


def central_difference(derivative_order, accuracy_order=None):
    num_points = 2 * ((derivative_order + accuracy_order - 1) // 2) + 1
    
    # Central stencil: symmetric around 0
    # [..., -2, -1, 0, 1, 2, ...]
    half = num_points // 2
    stencil = np.arange(-half, half + 1)
    
    return calculate_fd_coefficients(stencil, derivative_order)

# Calculate finite difference coefficients for a given derivative order
def get_fd_coefficients(derivative_order, forward_acc = 1, backward_acc = 1, central_acc = 2):    
    # Generate coefficients for each stencil type
    forward_coef, forward_power = forward_difference(derivative_order, forward_acc)
    central_coef, central_power = central_difference(derivative_order, central_acc)
    backward_coef, backward_power = backward_difference(derivative_order, backward_acc)
    
    coefficients = [
        np.round(forward_coef, decimals=10).tolist(),
        np.round(central_coef, decimals=10).tolist(),
        np.round(backward_coef, decimals=10).tolist()
    ]
    
    powers = (forward_power, central_power, backward_power)
    
    return coefficients, powers

def neumann_boundary_forward(h, accuracy_order=1):
    fd_coeffs, _ = forward_difference(derivative_order=1, accuracy_order=accuracy_order)

    # From: (c_0·f_0 + c_1·f_1 + c_2·f_2 + ...) / h = neumann_value
    # Solve: f_0 = (h·neumann_value - c_1·f_1 - c_2·f_2 - ...) / c_0
    c0 = fd_coeffs[0]
    neumann_scale = h / c0
    interior_coeffs = -fd_coeffs[1:] / c0
    
    return neumann_scale, interior_coeffs


def neumann_boundary_backward(h, accuracy_order=1):
    fd_coeffs, _ = backward_difference(derivative_order=1, accuracy_order=accuracy_order)

    # Reverse coefficients to match [0, -1, -2, ...] ordering
    fd_coeffs = fd_coeffs[::-1]

    # (c_0·f_N + c_1·f_{N-1} + c_2·f_{N-2} + ...) / h = neumann_value
    # f_N = (h·neumann_value - c_1·f_{N-1} - c_2·f_{N-2} - ...) / c_0
    c0 = fd_coeffs[0]
    neumann_scale = h / c0
    interior_coeffs = -fd_coeffs[1:] / c0

    return neumann_scale, interior_coeffs


if __name__ == "__main__":
    coeffs, powers = get_fd_coefficients(1)

    print(coeffs)
    print(powers)

    coeffs, powers = get_fd_coefficients(2)

    print(coeffs)
    print(powers)

    coeffs, powers = get_fd_coefficients(3)

    print(coeffs)
    print(powers)

    coeffs, powers = get_fd_coefficients(4)

    print(coeffs)
    print(powers)