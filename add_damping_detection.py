"""
Script to add damping detection and warnings to solver.py
"""

import re

# Read the current solver.py
with open('solver.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Replace the _has_damping method
old_has_damping = r'''    def _has_damping\(self\) -> tuple\[bool, float\]:
        """
        Detect if the equation has a damping term \(first-order time derivative in RHS\)\.

        For hyperbolic PDEs, checks if RHS contains ∂u/∂t term, which represents damping:
        ∂²u/∂t² = c²∇²u - γ\(∂u/∂t\)  <-- damping term

        Returns:
            tuple\[bool, float\]: \(has_damping, damping_coefficient\)
                - has_damping: True if damping term detected
                - damping_coefficient: The coefficient γ \(0\.0 if no damping\)
        """
        if self\.time_derivative_order != 2:
            return \(False, 0\.0\)

        # Check if RHS contains first-order time derivative
        import sympy as sp

        # The equation RHS is stored in self\.equation
        # We need to check if it contains diff\(u, t\)
        rhs = self\.equation\.rhs

        # Look for first-order time derivative
        first_time_deriv = sp\.diff\(self\.equation\.u_symbol, self\.equation\.t_symbol\)

        # Expand and check if this derivative appears
        rhs_expanded = sp\.expand\(rhs\)

        # Try to extract coefficient of ∂u/∂t
        if rhs\.has\(first_time_deriv\):
            # Extract coefficient \(may be spatially/temporally varying\)
            coeff = rhs_expanded\.coeff\(first_time_deriv\)

            if coeff is None:
                return \(False, 0\.0\)

            # Try to evaluate as constant
            # If it contains x, y, or t, it's not constant
            if coeff\.has\(self\.equation\.x_symbol\) or coeff\.has\(self\.equation\.y_symbol\) or coeff\.has\(self\.equation\.t_symbol\):
                # Spatially/temporally varying damping - return True but coefficient as NaN to indicate non-constant
                return \(True, float\('nan'\)\)

            # Constant damping coefficient
            try:
                gamma = float\(coeff\)
                return \(True, gamma\)
            except:
                return \(True, float\('nan'\)\)

        return \(False, 0\.0\)'''

new_has_damping = '''    def _has_damping(self) -> tuple[bool, float]:
        """
        Detect if the equation has a damping term (first-order time derivative in LHS).

        For hyperbolic PDEs, checks if LHS contains ∂u/∂t in addition to ∂²u/∂t²:
        ∂²u/∂t² + γ(∂u/∂t) = c²∇²u  <-- damping term on LHS

        Returns:
            tuple[bool, float]: (has_damping, damping_coefficient)
                - has_damping: True if damping term detected
                - damping_coefficient: The coefficient γ (0.0 if no damping, nan if non-constant)
        """
        if self.time_derivative_order != 2:
            return (False, 0.0)

        # Check if LHS contains first-order time derivative in addition to second-order
        import sympy as sp

        lhs = self.equation.lhs

        # Look for first-order time derivative
        first_time_deriv = sp.diff(self.equation.u_symbol, self.equation.t_symbol)

        # For damped equation: LHS = d²u/dt² + gamma*du/dt
        # Standard equation: LHS = d²u/dt²

        # Expand LHS
        lhs_expanded = sp.expand(lhs)

        # Check if first-order derivative appears
        if not lhs_expanded.has(first_time_deriv):
            return (False, 0.0)

        # Try to extract coefficient of ∂u/∂t
        try:
            # Collect terms with respect to the first derivative
            coeff = lhs_expanded.coeff(first_time_deriv)

            if coeff is None or coeff == 0:
                return (False, 0.0)

            # Check if coefficient is constant
            if coeff.has(self.equation.x_symbol) or coeff.has(self.equation.y_symbol) or coeff.has(self.equation.t_symbol):
                # Spatially/temporally varying damping
                return (True, float('nan'))

            # Constant damping coefficient
            gamma = float(coeff)
            if abs(gamma) > 1e-12:  # Non-zero damping
                return (True, gamma)
            else:
                return (False, 0.0)

        except:
            # If extraction fails, conservatively assume no damping
            return (False, 0.0)'''

# Apply replacement
if '_has_damping' in content:
    print("Replacing _has_damping method...")
    # Use a simpler marker-based replacement
    start_marker = '    def _has_damping(self)'
    end_marker = '    def _extract_wave_speed(self)'

    start_idx = content.find(start_marker)
    end_idx = content.find(end_marker)

    if start_idx != -1 and end_idx != -1:
        content = content[:start_idx] + new_has_damping + '\n\n' + content[end_idx:]
        print("✓ Replaced _has_damping")
    else:
        print("✗ Could not find markers")
else:
    print("_has_damping not found in file")

# Write back
with open('solver.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("\nDone! Now adding warnings to compute_momentum and compute_energy...")
