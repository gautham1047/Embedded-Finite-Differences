def drum_boundary_mask(grid, radius):
    """Returns mask for points outside drum (where BC should be applied)"""
    xv = grid.xv
    yv = grid.yv
    r_squared = xv**2 + yv**2
    return r_squared >= radius**2

from grid import Grid_2D

# Grid parameters
drum_radius = 1.0
x_i, x_f = (-drum_radius, drum_radius)
y_i, y_f = (-drum_radius, drum_radius)
t_i, t_f = (0, 5.0)

x_points = 40
y_points = 40
t_points = 300


grid = Grid_2D(x_points, y_points, x_i, x_f, y_i, y_f)

mask = drum_boundary_mask(grid, drum_radius)

import matplotlib.pyplot as plt

plt.imshow(mask, extent=(-1 * drum_radius, 1 * drum_radius, -1 * drum_radius, 1 * drum_radius))
plt.colorbar()
plt.title("Drum Boundary Mask")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()