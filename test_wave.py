import numpy as np
from sympy import symbols, diff, exp, pi
from grid import Grid_2D
from boundary_conditions import BoundaryConditions
from differential_equation import DifferentialEquation
from solver import Solver

# Define symbols
x, y, t = symbols('x y t')
from sympy import Function
u = Function('u')

# Grid parameters
x_i, x_f = (0, 1)
y_i, y_f = (0, 1)
t_i, t_f = (0, 0.5)

x_points = 20
y_points = 20
t_points = 100

c_squared = 1.0

pde_rhs = c_squared * (diff(u(x, y), x, x) + diff(u(x, y), y, y))

print("Creating differential equation...")
equation = DifferentialEquation(
    rhs=pde_rhs,
    u_symbol=u(x, y),
    x_symbol=x,
    y_symbol=y,
    t_symbol=t,
    time_derivative_order=2  # Hyperbolic PDE!
)

print(f"Equation: {equation}")
print(f"Is hyperbolic: {equation.is_hyperbolic}")

grid = Grid_2D(x_points, y_points, x_i, x_f, y_i, y_f)

bc = BoundaryConditions()

initial_u = exp(-50*((x-0.5)**2 + (y-0.5)**2))
initial_v = 0  

print("Creating solver...")
solver = Solver(
    equation=equation,
    grid=grid,
    boundary_conditions=bc,
    t_i=t_i, t_f=t_f, t_points=t_points,
    initial_condition=initial_u,
    initial_velocity=initial_v,
    accuracy_order=2,
    strategy='custom_stencil'
)

print("Solving with Newmark-beta method...")
solution = solver.solve_newmark()

print(f"\nSolution shape: {solution.shape}")
print(f"Initial max amplitude: {np.max(np.abs(solution[0])):.6f}")
print(f"Final max amplitude: {np.max(np.abs(solution[-1])):.6f}")

print("\nTesting velocity access...")
final_velocity = solver.get_velocity_at_time(-1)
print(f"Final velocity shape: {final_velocity.shape}")
print(f"Final max velocity: {np.max(np.abs(final_velocity)):.6f}")

final_velocity_2d = solver.get_velocity_2d(-1)
print(f"Final velocity 2D shape: {final_velocity_2d.shape}")

print("\nTest completed successfully!")

# Test energy conservation
print("\n" + "="*60)
print("ENERGY CONSERVATION VALIDATION")
print("="*60)

print("\nComputing conserved quantities...")
quantities = solver.compute_conserved_quantities()

print(f"Available quantities: {list(quantities.keys())}")

energy = quantities['energy']
momentum = quantities['momentum']

# Energy statistics
E_initial = energy[0]
E_final = energy[-1]
E_mean = np.mean(energy)
E_std = np.std(energy)
E_drift = (E_final - E_initial) / E_initial

print(f"\nEnergy Conservation:")
print(f"  Initial energy: {E_initial:.8e}")
print(f"  Final energy:   {E_final:.8e}")
print(f"  Mean energy:    {E_mean:.8e}")
print(f"  Std deviation:  {E_std:.8e}")
print(f"  Relative std:   {E_std / E_mean * 100:.4f}%")
print(f"  Energy drift:   {E_drift * 100:.4f}%")

if abs(E_drift) < 0.02:
    print(f"  PASS: Energy conserved to within {abs(E_drift) * 100:.4f}%")
else:
    print(f"  WARNING: Energy drift of {abs(E_drift) * 100:.4f}% detected")

# Momentum statistics
p_initial = momentum[0]
p_final = momentum[-1]
p_max = np.max(np.abs(momentum))

print(f"\nMomentum Conservation:")
print(f"  Initial momentum: {p_initial:.8e}")
print(f"  Final momentum:   {p_final:.8e}")
print(f"  Max |momentum|:   {p_max:.8e}")

if p_max < 1e-8:
    print(f"  PASS: Momentum conserved (max |p| < 1e-8)")
else:
    print(f"  WARNING: Non-zero momentum detected: {p_max:.8e}")

print("\nNext steps:")
print("1. Generate animation to visualize wave propagation")
print("2. Compare with analytical solution if available")
print("3. Run test_conservation.py for comprehensive validation tests")
