import numpy as np
from sympy import symbols, diff, exp, Function
from grid import Grid_2D
from boundary_conditions import DirichletMask
from differential_equation import DifferentialEquation
from solver import Solver

x, y, t = symbols('x y t')
u = Function('u')

c_squared = 1.0     
drum_radius = 1.0   

# Grid parameters
x_i, x_f = (-drum_radius, drum_radius)
y_i, y_f = (-drum_radius, drum_radius)
t_i, t_f = (0, 5.0)

x_points = 40
y_points = 40
t_points = 300

# d2u/dt2
lhs = diff(u(x, y), t, t)

# c^2 * (d2u/dx2 + d2u/dy2)
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

grid = Grid_2D(x_points, y_points, x_i, x_f, y_i, y_f,
               accuracy_order=2, strategy='custom_stencil')

def drum_boundary_mask(grid, radius):
    return grid.xv**2 + grid.yv**2 >= radius**2

bc = DirichletMask(
    mask_function=lambda g: drum_boundary_mask(g, drum_radius),
    dirichlet_value=0.0
)

initial_amplitude = 1.0
width = 0.3

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

h = grid.x_grid.delta
cfl = solver.t_delta / h
print(f"CFL = Δt/h = {cfl:.4f}  (must be < {1/np.sqrt(2):.4f} for stability)")
if cfl >= 1.0 / np.sqrt(2):
    print("bad CFL condition - Reduce t_points or increase grid resolution.")

# Solve using K-P leapfrog
solution = solver.solve_leapfrog(gamma=0.25)

print("animating...")

solver.animate('vibrating_drum/vibrating_drum_solution.gif')