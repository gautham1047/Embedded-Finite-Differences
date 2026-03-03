import numpy as np
from sympy import symbols, diff, exp, Function
from grid import VelocityGrid
from boundary_conditions import DirichletMask
from differential_equation import DifferentialEquation
from solver import Solver

# Define symbols
x, y, t = symbols('x y t')
u = Function('u')

# Physical parameters
c_squared = 1.0      
gamma = 0.5         
drum_radius = 1.0   

# Grid parameters
x_i, x_f = (-drum_radius, drum_radius)
y_i, y_f = (-drum_radius, drum_radius)
t_i, t_f = (0, 5.0)

x_points = 40
y_points = 40
t_points = 300

# d2u/dt2 + gamma·du/dt
lhs = diff(u(x, y), t, t) + gamma * diff(u(x, y), t)

# c²(d2u/dx2 + d2u/dy2)
rhs = c_squared * (diff(u(x, y), x, x) + diff(u(x, y), y, y))

equation = DifferentialEquation(
    rhs=rhs,
    lhs=lhs,
    u_symbol=u(x, y),
    x_symbol=x,
    y_symbol=y,
    t_symbol=t,
    time_derivative_order=2
)

# Create grid
grid = VelocityGrid(x_points, y_points, x_i, x_f, y_i, y_f, accuracy_order=2, strategy='custom_stencil')

# circular boundary condition
def drum_boundary_mask(grid, radius):
    xv = grid.xv
    yv = grid.yv
    r_squared = xv**2 + yv**2
    return r_squared >= radius**2

bc = DirichletMask(
    mask_function=lambda grid: drum_boundary_mask(grid, drum_radius),
    dirichlet_value=0.0
)

# initial condition
initial_amplitude = 1.0
width = 0.3  # Controls how sharp the bump is

initial_u = initial_amplitude * exp(-((x**2 + y**2) / (2 * width**2)))
initial_v = 0

solver = Solver(
    equation=equation,
    grid=grid,
    boundary_conditions=bc,
    t_i=t_i, t_f=t_f, t_points=t_points,
    initial_condition=initial_u,
    initial_velocity=initial_v,
)

solution = solver.solve_newmark(beta=0.25, gamma=0.5)

print("animating...")

solver.animate('vibrating_drum/vibrating_drum_damped_solution.gif')
solver.animate_velocity('vibrating_drum/vibrating_drum_damped_velocity.gif')