import numpy as np
from sympy.utilities.lambdify import lambdify
from typing import Union, Callable
import sympy as sp
from grid import Grid_2D
from boundary_conditions import BoundaryConditions
from differential_equation import DifferentialEquation
from animate import gen_anim, gen_velocity_anim

# Type alias for initial conditions
InitialConditionType = Union[sp.Expr, Callable[[np.ndarray, np.ndarray], np.ndarray], int, float]

class Solver:
    def __init__(self,
                 equation: DifferentialEquation,
                 grid: Grid_2D,
                 boundary_conditions: BoundaryConditions,
                 t_i: float,
                 t_f: float,
                 t_points: int,
                 initial_condition: InitialConditionType,
                 initial_velocity: InitialConditionType = None,
                 accuracy_order: int = 2,
                 strategy: str = 'custom_stencil'):
        # Store equation and basic parameters
        self.equation = equation
        self.grid = grid
        self.bc = boundary_conditions
        self.accuracy_order = accuracy_order
        self.strategy = strategy

        self.u_symbol = equation.u_symbol
        self.x_symbol = equation.x_symbol
        self.y_symbol = equation.y_symbol
        self.t_symbol = equation.t_symbol

        # Time discretization
        self.t_array = np.linspace(t_i, t_f, t_points)
        self.t_delta = (t_f - t_i) / (t_points - 1)
        self.t_points = t_points
        self.current_time = t_i
        self.current_step = 0

        self.solution_history = np.zeros((t_points, grid.x_points * grid.y_points))

        # PDE term storage
        self.derivative_matrices = {}       # Dict: (dx_order, dy_order) -> matrix
        self.coefficient_functions = {}     # Dict: (dx_order, dy_order) -> lambdified fn
        self.source_function = None         # Lambdified source term (or None)

        self.time_derivative_order = equation.time_derivative_order

        # Initialize components
        self.init_derivative_matrices(equation, grid, accuracy_order, strategy)
        self.init_grid(grid, initial_condition, t_i)
        self.init_velocity(grid, initial_velocity, t_points)

    def init_derivative_matrices(self, equation: DifferentialEquation, grid: Grid_2D,
                                 accuracy_order: int, strategy: str) -> None:
        term_dict = equation.extract_spatial_terms()

        # Build derivative matrices and lambdify coefficients
        for (dx_order, dy_order), coeff_expr in term_dict.items():
            if (dx_order, dy_order) == 'source':  # Skip source for now
                continue

            if dx_order > 0 and dy_order == 0:
                direction = 'x'
                order = dx_order
            elif dy_order > 0 and dx_order == 0:
                direction = 'y'
                order = dy_order
            elif dx_order > 0 and dy_order > 0:
                # Mixed derivative: d^m u / dx^m dy^n
                direction = 'xy'
                order = (dx_order, dy_order)
            else:
                # (0, 0) case - no matrix needed, just coefficient for u term
                direction = None
                order = 0

            # Create derivative matrix (if needed)
            if direction is not None:
                if direction == 'xy':
                    # Handle mixed derivatives: d^m u / dx^m dy^n = Dy^n @ Dx^m @ u
                    # Apply Dx first (innermost derivative), then Dy
                    Dx = grid.derivative_matrix(order=dx_order, direction='x',
                                               accuracy_order=accuracy_order, strategy=strategy)
                    Dy = grid.derivative_matrix(order=dy_order, direction='y',
                                               accuracy_order=accuracy_order, strategy=strategy)
                    matrix = Dy @ Dx
                else:
                    matrix = grid.derivative_matrix(order=order, direction=direction,
                                                   accuracy_order=accuracy_order,
                                                   strategy=strategy)
                self.derivative_matrices[(dx_order, dy_order)] = matrix

            # Coefficient function for each term
            coeff_func = lambdify([self.x_symbol, self.y_symbol, self.t_symbol], coeff_expr, modules='numpy')
            self.coefficient_functions[(dx_order, dy_order)] = coeff_func

        # Source term
        if 'source' in term_dict:
            source_expr = term_dict['source']
            self.source_function = lambdify([self.x_symbol, self.y_symbol, self.t_symbol], source_expr, modules='numpy')

    def init_grid(self, grid: Grid_2D, initial_condition: InitialConditionType, t_i: float) -> None:
        # Initialize grid
        if callable(initial_condition):
            # lambda function
            init_vals = initial_condition(grid.xv, grid.yv)
            grid.values = init_vals.flatten()
        else:
            # sympy expression
            grid.initialize_values(initial_condition, self.x_symbol, self.y_symbol)

        self.initial_values = grid.values.copy()
        self.t_i = t_i

        self.solution_history[0] = grid.values.copy()

    def init_velocity(self, grid: Grid_2D, initial_velocity: InitialConditionType,
                     t_points: int) -> None:
        # Initialize velocity for hyperbolic PDEs (time_derivative_order == 2)
        if self.time_derivative_order == 2:
            # Require initial velocity
            if initial_velocity is None:
                raise ValueError(
                    "initial_velocity required for hyperbolic PDEs (time_derivative_order=2). "
                    "Provide du/dt at t=0 as a SymPy expression, callable function, or scalar value."
                )

            # Initialize velocity using same pattern as initial_condition
            if callable(initial_velocity):
                # Callable function: initial_velocity(xv, yv) -> array
                init_vel = initial_velocity(grid.xv, grid.yv)
                self.velocity_values = init_vel.flatten()
            elif isinstance(initial_velocity, (int, float)):
                # Scalar value (e.g., 0) - apply uniformly
                self.velocity_values = np.full(grid.x_points * grid.y_points, initial_velocity)
            else:
                # SymPy expression: lambdify and evaluate
                vel_func = lambdify([self.x_symbol, self.y_symbol], initial_velocity, modules='numpy')
                init_vel = vel_func(grid.xv, grid.yv)
                self.velocity_values = init_vel.flatten()

            # Store initial velocity for reset()
            self.initial_velocity = self.velocity_values.copy()

            # Create velocity history storage
            self.velocity_history = np.zeros((t_points, grid.x_points * grid.y_points))
            self.velocity_history[0] = self.velocity_values.copy()

        elif self.time_derivative_order == 1:
            # Parabolic PDE - no velocity needed, save memory
            self.velocity_values = None
            self.initial_velocity = None
            self.velocity_history = None

        else:
            raise ValueError(
                f"Unsupported time_derivative_order={self.time_derivative_order}. "
                f"Must be 1 (parabolic) or 2 (hyperbolic)."
            )

    def compute_dudt(self, u_values: np.ndarray, time: float) -> np.ndarray:
        dudt = np.zeros_like(u_values)

        for (dx_order, dy_order), coeff_func in self.coefficient_functions.items():
            coeff_values = coeff_func(self.grid.xv, self.grid.yv, time)

            if np.isscalar(coeff_values):
                coeff = coeff_values
            else:
                coeff = coeff_values.flatten()

            if (dx_order, dy_order) == (0, 0):
                dudt += coeff * u_values
            else:
                matrix = self.derivative_matrices[(dx_order, dy_order)]
                dudt += coeff * (matrix @ u_values)

        if self.source_function is not None:
            source_values = self.source_function(self.grid.xv, self.grid.yv, time)
            if np.isscalar(source_values):
                dudt += source_values
            else:
                dudt += source_values.flatten()

        return dudt

    def apply_intermediate_bc(self, time: float) -> None:
        if self.bc.is_time_dependent:
            self.grid.apply_boundary_conditions(self.bc, accuracy_order=self.accuracy_order, time=time)

    def reset(self) -> None:
        self.grid.values = self.initial_values.copy()

        # Reset velocity for hyperbolic PDEs
        if self.time_derivative_order == 2:
            self.velocity_values = self.initial_velocity.copy()
            self.velocity_history = np.zeros((self.t_points, self.grid.x_points * self.grid.y_points))
            self.velocity_history[0] = self.initial_velocity.copy()

        self.current_time = self.t_i
        self.current_step = 0

        self.solution_history = np.zeros((self.t_points, self.grid.x_points * self.grid.y_points))
        self.solution_history[0] = self.initial_values.copy()

    def solve_euler(self) -> np.ndarray:
        if self.time_derivative_order != 1:
            raise ValueError(
                f"solve_euler() only supports first-order time derivatives (parabolic PDEs). "
                f"Current equation has time_derivative_order={self.time_derivative_order}. "
                f"Use solve_newmark() for second-order equations."
            )

        for step in range(1, self.t_points):
            self.current_step = step
            self.current_time = self.t_array[step - 1]

            dudt = self.compute_dudt(self.grid.values, self.current_time)
            self.grid.values += self.t_delta * dudt

            self.grid.apply_boundary_conditions(self.bc, accuracy_order=self.accuracy_order,
                                               time=self.t_array[step])

            self.solution_history[step] = self.grid.values.copy()

        self.current_time = self.t_array[-1]

        return self.solution_history

    def solve_rk4(self) -> np.ndarray:
        if self.time_derivative_order != 1:
            raise ValueError(
                f"solve_rk4() only supports first-order time derivatives (parabolic PDEs). "
                f"Current equation has time_derivative_order={self.time_derivative_order}. "
                f"Use solve_newmark() for second-order equations."
            )

        for step in range(1, self.t_points):
            self.current_step = step
            self.current_time = self.t_array[step - 1]

            u_n = self.grid.values.copy()

            k1 = self.compute_dudt(u_n, self.current_time)

            self.grid.values = u_n + 0.5 * self.t_delta * k1
            self.apply_intermediate_bc(self.current_time + 0.5 * self.t_delta)
            k2 = self.compute_dudt(self.grid.values, self.current_time + 0.5 * self.t_delta)

            self.grid.values = u_n + 0.5 * self.t_delta * k2
            self.apply_intermediate_bc(self.current_time + 0.5 * self.t_delta)
            k3 = self.compute_dudt(self.grid.values, self.current_time + 0.5 * self.t_delta)

            self.grid.values = u_n + self.t_delta * k3
            self.apply_intermediate_bc(self.current_time + self.t_delta)
            k4 = self.compute_dudt(self.grid.values, self.current_time + self.t_delta)

            self.grid.values = u_n + (self.t_delta / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            self.grid.apply_boundary_conditions(self.bc, accuracy_order=self.accuracy_order,
                                               time=self.t_array[step])

            self.solution_history[step] = self.grid.values.copy()

        self.current_time = self.t_array[-1]

        return self.solution_history

    def solve_newmark(self, beta: float = 0.25, gamma: float = 0.5) -> np.ndarray:
        if self.time_derivative_order != 2:
            raise ValueError(
                f"solve_newmark() only supports second-order time derivatives (hyperbolic PDEs). "
                f"Current equation has time_derivative_order={self.time_derivative_order}. "
                f"Use solve_euler() or solve_rk4() for first-order equations."
            )

        if not (0 <= beta <= 0.5):
            raise ValueError(f"Beta parameter {beta} outside stable range [0, 0.5]")
        if not (0 <= gamma <= 1):
            raise ValueError(f"Gamma parameter {gamma} outside valid range [0, 1]")

        u_n = self.grid.values.copy()
        v_n = self.velocity_values.copy()

        for step in range(1, self.t_points):
            self.current_step = step
            t_n = self.t_array[step - 1]
            t_np1 = self.t_array[step]

            a_n = self.compute_dudt(u_n, t_n)

            u_star = u_n + self.t_delta * v_n + (self.t_delta ** 2 / 2) * (1 - 2 * beta) * a_n
            v_star = v_n + self.t_delta * (1 - gamma) * a_n

            self.grid.values = u_star.copy()
            self.grid.apply_boundary_conditions(self.bc, accuracy_order=self.accuracy_order, time=t_np1)
            u_star = self.grid.values.copy()

            a_star = self.compute_dudt(u_star, t_np1)

            u_np1 = u_star + self.t_delta ** 2 * beta * a_star
            v_np1 = v_star + self.t_delta * gamma * a_star

            self.grid.values = u_np1.copy()
            self.grid.apply_boundary_conditions(self.bc, accuracy_order=self.accuracy_order, time=t_np1)
            u_np1 = self.grid.values.copy()

            self.solution_history[step] = u_np1.copy()
            self.velocity_history[step] = v_np1.copy()

            u_n = u_np1
            v_n = v_np1

        self.grid.values = u_n
        self.velocity_values = v_n
        self.current_time = self.t_array[-1]

        return self.solution_history

    def get_solution_at_time(self, time_index: int) -> np.ndarray:
        return self.solution_history[time_index]

    def get_solution_2d(self, time_index: int) -> np.ndarray:
        return self.solution_history[time_index].reshape((self.grid.y_points, self.grid.x_points))

    def get_velocity_at_time(self, time_index: int) -> np.ndarray:
        if self.velocity_history is None:
            raise ValueError(
                "Velocity history only available for hyperbolic PDEs (time_derivative_order=2). "
                "Current equation is parabolic (time_derivative_order=1)."
            )
        return self.velocity_history[time_index]

    def get_velocity_2d(self, time_index: int) -> np.ndarray:
        if self.velocity_history is None:
            raise ValueError(
                "Velocity history only available for hyperbolic PDEs (time_derivative_order=2). "
                "Current equation is parabolic (time_derivative_order=1)."
            )
        return self.velocity_history[time_index].reshape((self.grid.y_points, self.grid.x_points))

    def animate(self, file_name: str, z_label: str = "u", duration: float = 5.0) -> None:
        gen_anim(self.solution_history, self.grid, file_name, z_label, duration)

    def animate_velocity(self, file_name: str, duration: float = 5.0) -> None:
        if self.velocity_history is None:
            raise ValueError(
                "Velocity animation only available for hyperbolic PDEs (time_derivative_order=2). "
                "Current equation is parabolic (time_derivative_order=1)."
            )

        gen_velocity_anim(self.velocity_history, self.grid, file_name, duration)

    def _extract_wave_speed(self) -> float:
        """
        Extract wave speed squared (c²) from equation coefficients.

        Raises:
            ValueError: If no second-order spatial derivatives found
            ValueError: If wave speed differs in x and y directions
            ValueError: If wave speed is spatially or temporally varying
        """
        if self.time_derivative_order != 2:
            raise ValueError(
                f"_extract_wave_speed() only valid for hyperbolic PDEs (time_derivative_order=2). "
                f"Current equation has time_derivative_order={self.time_derivative_order}."
            )

        # Check for coefficient functions of second-order derivatives
        coeff_func_xx = self.coefficient_functions.get((2, 0))
        coeff_func_yy = self.coefficient_functions.get((0, 2))

        if coeff_func_xx is None or coeff_func_yy is None:
            raise ValueError(
                "Could not find second-order spatial derivative coefficients. "
                "Wave equation should have ∂²u/∂x² and ∂²u/∂y² terms."
            )

        # Evaluate at several points to check if constant
        test_points = [
            (self.grid.x_grid.x_i, self.grid.y_grid.x_i, self.t_i),
            (self.grid.x_grid.x_f, self.grid.y_grid.x_f, self.t_i),
            ((self.grid.x_grid.x_i + self.grid.x_grid.x_f) / 2,
             (self.grid.y_grid.x_i + self.grid.y_grid.x_f) / 2,
             self.t_i)
        ]

        c_squared_values_xx = []
        c_squared_values_yy = []

        for x_val, y_val, t_val in test_points:
            c_sq_xx = coeff_func_xx(x_val, y_val, t_val)
            c_sq_yy = coeff_func_yy(x_val, y_val, t_val)

            # Handle scalar or array returns
            if not np.isscalar(c_sq_xx):
                c_sq_xx = float(c_sq_xx.flatten()[0])
            if not np.isscalar(c_sq_yy):
                c_sq_yy = float(c_sq_yy.flatten()[0])

            c_squared_values_xx.append(c_sq_xx)
            c_squared_values_yy.append(c_sq_yy)

        # Check if constant across test points
        if not np.allclose(c_squared_values_xx, c_squared_values_xx[0], rtol=1e-10):
            raise ValueError(
                "Wave speed appears to be spatially or temporally varying. "
                "Energy calculation requires constant wave speed."
            )

        if not np.allclose(c_squared_values_yy, c_squared_values_yy[0], rtol=1e-10):
            raise ValueError(
                "Wave speed appears to be spatially or temporally varying. "
                "Energy calculation requires constant wave speed."
            )

        # Check if x and y wave speeds match
        c_squared_xx = c_squared_values_xx[0]
        c_squared_yy = c_squared_values_yy[0]

        if not np.isclose(c_squared_xx, c_squared_yy, rtol=1e-10):
            raise ValueError(
                f"Wave speed differs in x and y directions: c²_x={c_squared_xx}, c²_y={c_squared_yy}. "
                f"Energy calculation requires isotropic wave speed."
            )

        return float(c_squared_xx)

    def compute_total_content(self) -> np.ndarray:
        """
        Computes Q(t) = ∫∫ u(x,y,t) dx dy for all time steps.

        Conservation properties:
        - Conserved with zero-flux Neumann boundary conditions
        - Decays with Dirichlet boundary conditions (heat loss)
        """
        dx = self.grid.x_grid.delta
        dy = self.grid.y_grid.delta

        # Spatial area element
        dA = dx * dy

        # Compute integral at each time step
        Q = np.zeros(self.t_points)

        for step in range(self.t_points):
            u_2d = self.solution_history[step].reshape((self.grid.y_points, self.grid.x_points))
            # Rectangle rule integration
            Q[step] = np.sum(u_2d) * dA

        return Q

    def compute_momentum(self) -> np.ndarray:
        """
        Computes p(t) = ∫∫ (∂u/∂t)(x,y,t) dx dy for all time steps.

        Conservation properties:
        - Conserved with zero Dirichlet boundary conditions and symmetric setup
        - Should remain constant for isolated wave systems
        """
        if self.time_derivative_order != 2:
            raise ValueError(
                f"Momentum only defined for hyperbolic PDEs (time_derivative_order=2). "
                f"Current equation has time_derivative_order={self.time_derivative_order}. "
                f"This is a parabolic PDE."
            )

        dx = self.grid.x_grid.delta
        dy = self.grid.y_grid.delta
        dA = dx * dy

        # Compute momentum at each time step
        p = np.zeros(self.t_points)

        for step in range(self.t_points):
            v_2d = self.velocity_history[step].reshape((self.grid.y_points, self.grid.x_points))
            # Integrate velocity field
            p[step] = np.sum(v_2d) * dA

        return p

    def compute_energy(self) -> np.ndarray:
        """
        Compute total energy for hyperbolic PDEs (wave equation).

        Computes E(t) = KE(t) + PE(t) where:
        - Kinetic energy: KE = ∫∫ (1/2)(∂u/∂t)² dx dy
        - Potential energy: PE = ∫∫ (c²/2)(∇u)² dx dy

        For wave equation with appropriate boundary conditions, total energy
        should be conserved (constant over time).
        """
        if self.time_derivative_order != 2:
            raise ValueError(
                f"Energy only defined for hyperbolic PDEs (time_derivative_order=2). "
                f"Current equation has time_derivative_order={self.time_derivative_order}. "
                f"This is a parabolic PDE."
            )

        # Extract wave speed
        c_squared = self._extract_wave_speed()

        # Ensure we have gradient derivative matrices
        if (1, 0) not in self.derivative_matrices:
            # Compute first derivative in x on demand
            self.derivative_matrices[(1, 0)] = self.grid.derivative_matrix(
                order=1, direction='x',
                accuracy_order=self.accuracy_order,
                strategy=self.strategy
            )

        if (0, 1) not in self.derivative_matrices:
            # Compute first derivative in y on demand
            self.derivative_matrices[(0, 1)] = self.grid.derivative_matrix(
                order=1, direction='y',
                accuracy_order=self.accuracy_order,
                strategy=self.strategy
            )

        dx = self.grid.x_grid.delta
        dy = self.grid.y_grid.delta
        dA = dx * dy

        D_x = self.derivative_matrices[(1, 0)]
        D_y = self.derivative_matrices[(0, 1)]

        # Compute energy at each time step
        E = np.zeros(self.t_points)

        for step in range(self.t_points):
            # Kinetic energy: (1/2) * ∫∫ v² dx dy
            v_2d = self.velocity_history[step].reshape((self.grid.y_points, self.grid.x_points))
            KE = 0.5 * np.sum(v_2d**2) * dA

            # Potential energy: (c²/2) * ∫∫ (∇u)² dx dy
            u_flat = self.solution_history[step]

            # Compute spatial gradients
            du_dx = D_x @ u_flat
            du_dy = D_y @ u_flat

            # |∇u|² = (∂u/∂x)² + (∂u/∂y)²
            grad_u_squared = du_dx**2 + du_dy**2
            grad_u_squared_2d = grad_u_squared.reshape((self.grid.y_points, self.grid.x_points))

            PE = 0.5 * c_squared * np.sum(grad_u_squared_2d) * dA

            E[step] = KE + PE

        return E

    def compute_conserved_quantities(self) -> dict:
        """
        Auto-detect PDE type and compute appropriate conserved quantities.

        For hyperbolic PDEs (wave equation, time_derivative_order=2):
            - 'energy': Total energy (kinetic + potential)
            - 'momentum': Total momentum
            - 'total_content': Total displacement integral

        For parabolic PDEs (heat equation, time_derivative_order=1):
            - 'total_content': Total heat/mass content

        Returns:
            dict: Dictionary mapping quantity names to time-series array
        """
        if self.time_derivative_order == 2:
            # Hyperbolic PDE (wave equation)
            return {
                'energy': self.compute_energy(),
                'momentum': self.compute_momentum(),
                'total_content': self.compute_total_content()
            }
        elif self.time_derivative_order == 1:
            # Parabolic PDE (heat/diffusion equation)
            return {
                'total_content': self.compute_total_content()
            }
        else:
            raise ValueError(
                f"Unsupported time_derivative_order={self.time_derivative_order}. "
                f"Must be 1 (parabolic) or 2 (hyperbolic)."
            )