"""
Generates an approximate solution to the 2D convection-diffusion equation using finite differences.

The equation: dT/dt = a * (d2T/dx2 + d2T/dy2) - a * (u_x * dT/dx + u_y * dT/dy) + Q / cp

When running this program, you may have to create a folder named 'tmp' to store the graphs.
"""

import numpy as np
from sympy import exp, symbols, sqrt
from sympy.utilities.lambdify import lambdify

from grid import Grid_2D
from animate import gen_anim

np.set_printoptions(linewidth=np.inf)

# Domain and time range
x_i, x_f = (0, 1)
y_i, y_f = (0, 1)
t_i, t_f = (0, 1)

L_x = x_f - x_i
L_y = y_f - y_i

x_points = 10
y_points = 10
t_points = 500

# Create grid
grid = Grid_2D(x_points, y_points, x_i, x_f, y_i, y_f)

t_delta = (t_f - t_i) / (t_points - 1)
t = np.linspace(t_i, t_f, t_points)

# Sympy symbols
x_symbol, y_symbol, t_symbol = symbols('x y t')

# Convection-Diffusion equation parameters
# dT/dt = a * d2T/dx2 - a * u * dT/dx + Q / cp
a = -0.0025  # Diffusion coefficient
e = 1  # ε (epsilon)
Q_const = 0  # Heat source
c = 1  # cp (specific heat capacity)

# Velocity field components (normalized circular flow around center)
u_x_expr = -2 * (y_symbol - L_y / 2)
u_y_expr = 2 * (x_symbol - L_x / 2)

# Normalize velocity field
u_x_expr /= sqrt((x_symbol - L_x / 2.0) ** 2.0 + (y_symbol - L_y / 2.0) ** 2.0)
u_y_expr /= sqrt((x_symbol - L_x / 2.0) ** 2.0 + (y_symbol - L_y / 2.0) ** 2.0)

# Lambdify velocity field and source term
u_x = np.vectorize(lambdify([x_symbol, y_symbol, t_symbol], u_x_expr))
u_y = np.vectorize(lambdify([x_symbol, y_symbol, t_symbol], u_y_expr))
Q = np.vectorize(lambdify([x_symbol, y_symbol, t_symbol], Q_const))

# Initial condition 
scale_factor = 10
alpha = 70
f_0 = exp(-alpha * ((x_symbol - L_x / 2.0) ** 2.0 + (y_symbol - L_y / 4.0) ** 2.0))
grid.initialize_values(f_0, x_symbol, y_symbol)

# Boundary conditions
T_1 = 0
T_2 = 0
y_is_dirichlet = True

# Boundary condition functions
x_0_func = lambda y: np.zeros_like(y) + T_1  # x = x_i boundary
x_L_func = lambda y: np.zeros_like(y) + T_2  # x = x_f boundary

if y_is_dirichlet:
    # Dirichlet: linear interpolation between T_1 and T_2
    y_0_func = lambda x: T_1 + (T_2 - T_1) * x / L_x  # y = y_i boundary
    y_L_func = lambda x: T_1 + (T_2 - T_1) * x / L_x  # y = y_f boundary
else:
    # Neumann: zero derivative
    y_0_func = lambda x: np.zeros_like(x)
    y_L_func = lambda x: np.zeros_like(x)

# Create derivative matrices using Grid_2D methods
# Laplacian matrix (second derivatives): ∂²/∂x² + ∂²/∂y²
A = grid.derivative_matrix(order=2, direction='xy')

# First derivative matrices for convection terms
C_x = grid.derivative_matrix(order=1, direction='x')  # ∂/∂x
C_y = grid.derivative_matrix(order=1, direction='y')  # ∂/∂y

def apply_boundary_conditions(u: np.ndarray) -> np.ndarray:
    """Apply boundary conditions to the solution vector."""
    if y_is_dirichlet:
        # Dirichlet boundary conditions
        u[0:x_points] = y_0_func(grid.x_grid.x)
        u[-x_points:] = y_L_func(grid.x_grid.x)
    else:
        # Neumann boundary conditions (using second-order finite differences)
        # Bottom boundary (y = y_i)
        index = 0
        for y_deriv in y_0_func(grid.x_grid.x):
            u[index] = (4 * u[index + x_points] - u[index + 2 * x_points] +
                       y_deriv * grid.y_grid.delta * 2) / 3
            index += 1

        # Top boundary (y = y_f)
        index = (y_points - 1) * x_points
        for y_deriv in y_L_func(grid.x_grid.x):
            u[index] = (4 * u[index - x_points] - u[index - 2 * x_points] +
                       y_deriv * grid.y_grid.delta * 2) / 3
            index += 1

    # Left and right boundaries (x = x_i and x = x_f)
    index = 0
    for x_val in x_0_func(grid.y_grid.x):
        u[index] = x_val
        index += x_points

    index = x_points - 1
    for x_val in x_L_func(grid.y_grid.x):
        u[index] = x_val
        index += x_points

    return u

# Storage for solution at each time step
u_mat = np.zeros((t_points, x_points * y_points))
u_mat[0] = grid.values.copy()

# Time evolution using finite differences
u = grid.values.copy()

print(f"Stability parameter: {t_delta / (grid.x_grid.delta ** 2) + t_delta / (grid.y_grid.delta ** 2)}")

for index, curr_t in enumerate(t[:-1], start=1):
    # Calculate all derivative terms
    d_2 = a * (A @ u)  # Diffusion term
    d_1_x = e * u_x(grid.xv, grid.yv, curr_t).flatten() * (C_x @ u)  # Convection in x
    d_1_y = e * u_y(grid.xv, grid.yv, curr_t).flatten() * (C_y @ u)  # Convection in y
    d_0 = Q(grid.xv, grid.yv, curr_t).flatten() / c  # Source term

    # Update solution: du/dt = d_2 - d_1_x - d_1_y + d_0
    u += t_delta * (d_2 - d_1_x - d_1_y + d_0)

    # Apply boundary conditions
    u = apply_boundary_conditions(u)

    # Store solution
    u_mat[index] = u.copy()

# Generate animation
gen_anim(u_mat, grid.xv, grid.yv, x_i, x_f, y_i, y_f, 'tmp/approximate_solution_2d.gif')

print("done")
