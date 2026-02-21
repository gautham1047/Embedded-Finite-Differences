"""
Test the Solver class with the heat equation.

Steady-state solution: u(x,y) = x (plane from z=0 at x=0 to z=1 at x=1)

Boundary conditions:
- Dirichlet at x=0: u = 0
- Dirichlet at x=1: u = 1
- Neumann at y=0: du/dy = 0
- Neumann at y=1: du/dy = 0
"""

import numpy as np
from sympy import symbols, diff, sin, pi, Function
from grid import Grid_2D
from boundary_conditions import BoundaryConditions
from differential_equation import DifferentialEquation
from solver import Solver
from animate import gen_anim

# Define symbols
x, y, t = symbols('x y t')
u = Function('u')

# Grid parameters
x_i, x_f = (0, 1)
y_i, y_f = (0, 1)
t_i, t_f = (0, 5)  # Longer time to reach steady state

x_points = 20
y_points = 20
t_points = 500

# Heat equation parameters
a = 0.1

# Define PDE: du/dt = a * (d2u/dx2 + d2u/dy2)
pde_rhs = a * (diff(u(x, y), x, x) + diff(u(x, y), y, y))

# Create differential equation object
print("Creating differential equation...")
equation = DifferentialEquation(
    rhs=pde_rhs,
    u_symbol=u(x, y),
    x_symbol=x,
    y_symbol=y,
    t_symbol=t,
    time_derivative_order=1  # First-order in time (parabolic PDE)
)

print(f"Equation: {equation}")
print(f"Is parabolic: {equation.is_parabolic}")

# Create grid
grid = Grid_2D(x_points, y_points, x_i, x_f, y_i, y_f)

# Define boundary conditions:
# - Dirichlet at x=0: u = 0
# - Dirichlet at x=1: u = 1
# - Neumann at y=0 and y=1: du/dy = 0 (zero flux)
bc = BoundaryConditions(
    x_0_func=lambda y: 0*y,                # u = 0 at x = 0
    x_L_func=lambda y: 0*y + 1,            # u = 1 at x = 1
    y_0_func=lambda x: 0*x,                # du/dy = 0 at y = 0 (Neumann)
    y_L_func=lambda x: 0*x,                # du/dy = 0 at y = 1 (Neumann)
    x_0_is_dirichlet=True,
    x_L_is_dirichlet=True,
    y_0_is_dirichlet=False,      # Neumann
    y_L_is_dirichlet=False,      # Neumann
)

# Define initial condition: start with a sinusoidal perturbation
# This will evolve to the steady-state u = x
initial = x + 0.5 * sin(pi * x) * sin(pi * y)

# Create solver
print("Creating solver...")
solver = Solver(
    equation=equation,
    grid=grid,
    boundary_conditions=bc,
    t_i=t_i, t_f=t_f, t_points=t_points,
    initial_condition=initial,
    accuracy_order=2,
    strategy='custom_stencil'
)

print("Solving...")
solution_1 = solver.solve_euler()
solver.animate('heat2/solver_test_euler.gif')

solver.reset()
solution_2 = solver.solve_rk4()
solver.animate('heat2/solver_test_rk4.gif')

print("Done!")
