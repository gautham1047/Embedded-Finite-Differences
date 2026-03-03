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


def dirichlet_pseudo_boundary_forward(alpha_over_h, derivative_order, n):
    """
    FD coefficients for a derivative at a pseudo-boundary point whose Dirichlet
    boundary lies in the + direction at fractional grid distance alpha_over_h.

    Stencil (in grid units): {-n, -(n-1), ..., -1, 0, alpha_over_h}
    The first n+1 entries are interior grid points; the last is the boundary.

    Parameters
    ----------
    alpha_over_h   : physical distance to boundary / grid spacing  (> 0)
    derivative_order : order of derivative to approximate
    n              : number of interior points used in the - direction
                     (offsets {-n, ..., 0}), so the stencil has n+2 points total.
                     Must satisfy n+1 > derivative_order.

    Returns
    -------
    interior_coeffs : ndarray, shape (n+1,)
        Raw FD coefficients for offsets {-n, ..., -1, 0}.
    boundary_coeff  : float
        Raw FD coefficient for the Dirichlet boundary point.

    Derivative ≈ (1/h^d) * (interior_coeffs @ [f_{i-n},...,f_i] + boundary_coeff * dirichlet_val)
    """
    interior_offsets = np.arange(-n, 1, dtype=float)          # {-n, ..., 0}
    stencil = np.append(interior_offsets, float(alpha_over_h))
    coeffs, _ = calculate_fd_coefficients(stencil, derivative_order)
    return coeffs[:-1], coeffs[-1]


def dirichlet_pseudo_boundary_backward(alpha_over_h, derivative_order, n):
    """
    FD coefficients for a derivative at a pseudo-boundary point whose Dirichlet
    boundary lies in the - direction at fractional grid distance alpha_over_h.

    Stencil (in grid units): {-alpha_over_h, 0, 1, ..., n}
    The first entry is the boundary; the remaining n+1 are interior grid points.

    Parameters
    ----------
    alpha_over_h   : physical distance to boundary / grid spacing  (> 0)
    derivative_order : order of derivative to approximate
    n              : number of interior points used in the + direction
                     (offsets {0, ..., n}), so the stencil has n+2 points total.
                     Must satisfy n+1 > derivative_order.

    Returns
    -------
    interior_coeffs : ndarray, shape (n+1,)
        Raw FD coefficients for offsets {0, 1, ..., n}.
    boundary_coeff  : float
        Raw FD coefficient for the Dirichlet boundary point.

    Derivative ≈ (1/h^d) * (boundary_coeff * dirichlet_val + interior_coeffs @ [f_i,...,f_{i+n}])
    """
    interior_offsets = np.arange(0, n + 1, dtype=float)          # {0, ..., n}
    stencil = np.append(-float(alpha_over_h), interior_offsets)
    coeffs, _ = calculate_fd_coefficients(stencil, derivative_order)
    return coeffs[1:], coeffs[0]


def lagrange_weights(x_nodes: np.ndarray, x_query: float) -> np.ndarray:
    """
    Quadratic Lagrange interpolation weights at x_query given 3 node positions.

    Parameters
    ----------
    x_nodes : array-like of shape (3,), the node x-coordinates (need not be uniform)
    x_query : scalar, the evaluation point

    Returns
    -------
    weights : ndarray of shape (3,), such that
              interpolated_value ≈ weights @ f_values
    """
    x0, x1, x2 = x_nodes
    w0 = (x_query - x1) * (x_query - x2) / ((x0 - x1) * (x0 - x2))
    w1 = (x_query - x0) * (x_query - x2) / ((x1 - x0) * (x1 - x2))
    w2 = (x_query - x0) * (x_query - x1) / ((x2 - x0) * (x2 - x1))
    return np.array([w0, w1, w2])


def normal_lagrange_weights(xi_gamma: float, xi_I: float) -> tuple:
    """
    Lagrange weights along the inward normal for the K-P ghost point formula.

    Nodes are at distances 0 (ghost point), xi_I, and 2*xi_I from the ghost point.
    Evaluates the Lagrange basis at xi_gamma (distance from ghost to boundary).

    Parameters
    ----------
    xi_gamma : distance from ghost point to the boundary along the normal (0 < xi_gamma <= xi_I)
    xi_I     : distance from ghost point to the first interior grid-line crossing

    Returns
    -------
    (g0, gI, gII) : Lagrange weights at positions 0, xi_I, 2*xi_I
    """
    xi_II = 2.0 * xi_I
    g0  = (xi_gamma - xi_I)  * (xi_gamma - xi_II) / ((-xi_I) * (-xi_II))
    gI  =  xi_gamma           * (xi_gamma - xi_II) / (xi_I   * (xi_I - xi_II))
    gII =  xi_gamma           * (xi_gamma - xi_I)  / (xi_II  * (xi_II - xi_I))
    return g0, gI, gII


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