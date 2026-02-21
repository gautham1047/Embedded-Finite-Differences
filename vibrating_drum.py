import numpy as np
from sympy import symbols, diff, exp, Function
from grid import Grid_2D
from boundary_conditions import DirichletMask
from differential_equation import DifferentialEquation
from solver import Solver
import matplotlib.pyplot as plt

print("="*70)
print("VIBRATING DRUM")
print("="*70)

# Define symbols
x, y, t = symbols('x y t')
u = Function('u')

# Physical parameters
c_squared = 1.0      # Wave speed squared (tension/density ratio)
drum_radius = 1.0    # Drum radius

print(f"\nPhysical Parameters:")
print(f"  Wave speed c: {np.sqrt(c_squared):.2f}")
print(f"  Drum radius: {drum_radius:.2f}")

# Grid parameters
x_i, x_f = (-drum_radius, drum_radius)
y_i, y_f = (-drum_radius, drum_radius)
t_i, t_f = (0, 5.0)

x_points = 40
y_points = 40
t_points = 300

print(f"\nGrid Parameters:")
print(f"  Spatial domain: [{x_i}, {x_f}] × [{y_i}, {y_f}]")
print(f"  Grid points: {x_points} × {y_points}")
print(f"  Time domain: [{t_i}, {t_f}]")
print(f"  Time steps: {t_points}")

# LHS d2u/dt2 + gamma·du/dt
lhs = diff(u(x, y), t, t) 

# RHS: c^2 * (d2u/dx2 + d2u/dy2)
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

print(f"  LHS: {equation.lhs}")
print(f"  RHS: {equation.rhs}")
print(f"  Is hyperbolic: {equation.is_hyperbolic}")

# Create grid
grid = Grid_2D(x_points, y_points, x_i, x_f, y_i, y_f)

# Boundary conditions - fixed drum edge
# For circular drum, we need to set u=0 where x^2+y^2 >= R^2
def drum_boundary_mask(grid, radius):
    """Returns mask for points outside drum (where BC should be applied)"""
    xv = grid.xv
    yv = grid.yv
    r_squared = xv**2 + yv**2
    return r_squared >= radius**2

# Create DirichletMask boundary condition for the circular drum
# This will be applied automatically during time evolution
bc = DirichletMask(
    mask_function=lambda grid: drum_boundary_mask(grid, drum_radius),
    dirichlet_value=0.0
)

# Initial condition: Gaussian bump at center
# Amplitude decreases with distance from center
initial_amplitude = 1.0
width = 0.3  # Controls how sharp the bump is

initial_u = initial_amplitude * exp(-((x**2 + y**2) / (2 * width**2)))

# Initial velocity: zero (released from rest)
initial_v = 0

print(f"\nInitial Conditions:")
print(f"  Displacement: Gaussian bump, amplitude={initial_amplitude}, width={width}")
print(f"  Velocity: 0 (released from rest)")

# Create solver
print(f"\nCreating solver...")
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

print(f"  Solver created successfully")
print(f"  Time step Δt: {solver.t_delta:.6f}")

# Solve using Newmark-beta method
print(f"\nSolving wave equation with Newmark-beta method...")
print(f"  (This may take a moment...)")
print(f"  Circular drum boundary will be enforced automatically during evolution")

solution = solver.solve_newmark(beta=0.25, gamma=0.5)

print(f"  Solution computed!")

# Analyze the solution
print(f"\n" + "="*70)
print("SOLUTION ANALYSIS")
print("="*70)

# Basic statistics
initial_max = np.max(np.abs(solution[0]))
final_max = np.max(np.abs(solution[-1]))
decay_ratio = final_max / initial_max

print(f"\nAmplitude Statistics:")
print(f"  Initial max amplitude: {initial_max:.6f}")
print(f"  Final max amplitude:   {final_max:.6f}")
print(f"  Decay ratio:           {decay_ratio:.6f} ({(1-decay_ratio)*100:.1f}% reduction)")

# Compute conserved quantities (with damping warnings expected)
print(f"\n" + "-"*70)
print("Computing Energy and Momentum (damping warnings expected)...")
print("-"*70)

quantities = solver.compute_conserved_quantities()

energy = quantities['energy']
momentum = quantities['momentum']
content = quantities['total_content']

# Energy analysis
E_initial = energy[0]
E_final = energy[-1]
E_decay = (E_initial - E_final) / E_initial

print(f"\nEnergy Analysis:")
print(f"  Initial energy: {E_initial:.8e}")
print(f"  Final energy:   {E_final:.8e}")
print(f"  Energy decay:   {E_decay * 100:.2f}%")

# Momentum analysis
p_max = np.max(np.abs(momentum))

print(f"\nMomentum Analysis:")
print(f"  Max |momentum|: {p_max:.8e}")
print(f"  Expected: Should be ~0 (symmetric initial condition)")

# Plot results
print(f"\n" + "="*70)
print("GENERATING PLOTS")
print("="*70)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle(f'Vibrating Drum', fontsize=16)

# Row 1: Displacement at different times
times_to_plot = [0, t_points//4, t_points//2, -1]
time_labels = ['t=0', f't={solver.t_array[t_points//4]:.2f}',
               f't={solver.t_array[t_points//2]:.2f}', f't={solver.t_array[-1]:.2f}']

for idx, (step, label) in enumerate(zip(times_to_plot[:3], time_labels[:3])):
    ax = axes[0, idx]
    u_2d = solver.get_solution_2d(step)

    # Mask outside drum for visualization
    mask = drum_boundary_mask(grid, drum_radius)
    u_2d_masked = np.copy(u_2d)
    u_2d_masked[mask] = np.nan

    im = ax.contourf(grid.xv, grid.yv, u_2d_masked, levels=20, cmap='RdBu_r')
    ax.set_title(label)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, label='Displacement')

    # Draw drum boundary
    circle = plt.Circle((0, 0), drum_radius, fill=False, color='black', linewidth=2)
    ax.add_patch(circle)

# Row 2, Col 1: Energy over time
ax = axes[1, 0]
ax.plot(solver.t_array, energy, 'b-', linewidth=2)
ax.set_xlabel('Time')
ax.set_ylabel('Total Energy')
ax.set_title('Energy Dissipation')
ax.grid(True, alpha=0.3)
ax.text(0.05, 0.95, f'Decay: {E_decay*100:.1f}%',
        transform=ax.transAxes, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Row 2, Col 2: Momentum over time
ax = axes[1, 1]
ax.plot(solver.t_array, momentum, 'g-', linewidth=2)
ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax.set_xlabel('Time')
ax.set_ylabel('Momentum')
ax.set_title('Momentum Conservation')
ax.grid(True, alpha=0.3)

# Row 2, Col 3: Max amplitude over time
ax = axes[1, 2]
max_amplitude = np.array([np.max(np.abs(solver.solution_history[step]))
                          for step in range(t_points)])
ax.plot(solver.t_array, max_amplitude, 'r-', linewidth=2)
ax.set_xlabel('Time')
ax.set_ylabel('Max |u|')
ax.set_title('Amplitude Decay')
ax.grid(True, alpha=0.3)


plt.tight_layout()
plt.savefig('vibrating_drum/vibrating_drum.png', dpi=150, bbox_inches='tight')
print(f"\n  Saved plot: vibrating_drum/vibrating_drum.png")

solver.animate('vibrating_drum/vibrating_drum_solution.gif')
solver.animate_velocity('vibrating_drum/vibrating_drum_velocity.gif')

print(f"\n" + "="*70)
print("DONE!")
print("="*70)
