from typing import Callable, Optional
import numpy as np
import inspect

class DirichletMask:
    def __init__(self, mask_function: Callable, dirichlet_value: float = 0.0):
        self.mask_function = mask_function
        self.dirichlet_value = dirichlet_value
        self._cached_mask = None
        self._cached_grid_shape = None

    def get_mask(self, grid):
        # Cache the mask to avoid recomputing it at every time step
        grid_shape = (grid.x_points, grid.y_points)
        if self._cached_mask is None or self._cached_grid_shape != grid_shape:
            self._cached_mask = self.mask_function(grid)
            self._cached_grid_shape = grid_shape
        return self._cached_mask

class BoundaryConditions:
    def __init__(
        self,
        x_0_func: Optional[Callable] = None,
        x_L_func: Optional[Callable] = None,
        y_0_func: Optional[Callable] = None,
        y_L_func: Optional[Callable] = None,
        x_0_is_dirichlet: bool = True,
        x_L_is_dirichlet: bool = True,
        y_0_is_dirichlet: bool = True,
        y_L_is_dirichlet: bool = True,
    ):
        # Set default functions (zero boundary)
        default_func = lambda coords: np.zeros_like(coords)

        self.x_0_func = x_0_func if x_0_func is not None else default_func
        self.x_L_func = x_L_func if x_L_func is not None else default_func
        self.y_0_func = y_0_func if y_0_func is not None else default_func
        self.y_L_func = y_L_func if y_L_func is not None else default_func

        self.x_0_is_dirichlet = x_0_is_dirichlet
        self.x_L_is_dirichlet = x_L_is_dirichlet
        self.y_0_is_dirichlet = y_0_is_dirichlet
        self.y_L_is_dirichlet = y_L_is_dirichlet

        # Detect time dependency by checking function signatures
        self.is_time_dependent = self._check_time_dependency()

    def _check_time_dependency(self) -> bool:
        """
        Check if any boundary function accepts a time parameter.

        Returns True if any function has a signature accepting 2+ parameters
        (spatial_coords, time) or has 'time' in parameter names.
        """
        all_funcs = [self.x_0_func, self.x_L_func, self.y_0_func, self.y_L_func]

        for func in all_funcs:
            try:
                sig = inspect.signature(func)
                params = list(sig.parameters.keys())

                # Check if function accepts 2+ parameters or has 'time' parameter
                if len(params) >= 2 or 'time' in params:
                    return True
            except (ValueError, TypeError):
                # Can't inspect (e.g., built-in function), assume time-independent
                continue

        return False

    @property
    def dirichlet_boundaries(self):
        boundaries = {}
        if self.x_0_is_dirichlet:
            boundaries['x_0'] = self.x_0_func
        if self.x_L_is_dirichlet:
            boundaries['x_L'] = self.x_L_func
        if self.y_0_is_dirichlet:
            boundaries['y_0'] = self.y_0_func
        if self.y_L_is_dirichlet:
            boundaries['y_L'] = self.y_L_func
        return boundaries

    @property
    def neumann_boundaries(self):
        boundaries = {}
        if not self.x_0_is_dirichlet:
            boundaries['x_0'] = self.x_0_func
        if not self.x_L_is_dirichlet:
            boundaries['x_L'] = self.x_L_func
        if not self.y_0_is_dirichlet:
            boundaries['y_0'] = self.y_0_func
        if not self.y_L_is_dirichlet:
            boundaries['y_L'] = self.y_L_func
        return boundaries

