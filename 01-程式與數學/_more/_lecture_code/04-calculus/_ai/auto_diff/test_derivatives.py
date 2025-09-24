"""
Pytest test to compare numerical and automatic differentiation results.
"""

import pytest
import torch
import math
from derivatives import numerical_derivative, automatic_derivative

# A small tolerance for floating-point comparisons.
# The value is set to account for the approximation error in numerical differentiation.
TOLERANCE = 1e-4

def create_tensors(value):
    """Helper function to create a PyTorch tensor with requires_grad=True."""
    return torch.tensor(value, requires_grad=True)

# Test case 1: f(x) = x^2
def test_derivative_of_x_squared_at_3():
    """
    Tests the derivative of f(x) = x^2 at x = 3.
    """
    def f(x):
        return x**2

    x_val = 3.0
    x_tensor = create_tensors(x_val)

    deriv_numerical = numerical_derivative(f, x_val)
    deriv_automatic = automatic_derivative(f, x_tensor)

    assert math.isclose(deriv_numerical, deriv_automatic, rel_tol=TOLERANCE), \
        f"Mismatch for f(x)=x^2: Numerical={deriv_numerical}, Automatic={deriv_automatic}"

# Test case 2: f(x) = sin(x)
def test_derivative_of_sin_x_at_pi_over_2():
    """
    Tests the derivative of f(x) = sin(x) at x = pi/2.
    """
    def f_tensor(x_tensor):
        return torch.sin(x_tensor)
        
    def f_float(x_val):
        return math.sin(x_val)

    x_val = math.pi / 2
    x_tensor = create_tensors(x_val)

    deriv_numerical = numerical_derivative(f_float, x_val)
    deriv_automatic = automatic_derivative(f_tensor, x_tensor)
    
    # Reset gradient for the next test
    x_tensor.grad.zero_()

    # 注意，以下 isclose 要設定 abs_tol，因為 sin(x) 在 pi/2 點的導數是 0，rel_tol 對於接近 0 的值不適用
    assert math.isclose(deriv_numerical, deriv_automatic, rel_tol=TOLERANCE, abs_tol=TOLERANCE), \
        f"Mismatch for f(x)=sin(x): Numerical={deriv_numerical}, Automatic={deriv_automatic}"

# Test case 3: A polynomial function
def test_derivative_of_polynomial_at_2():
    """
    Tests the derivative of f(x) = 3x^3 - 2x + 1 at x = 2.
    """
    def f_tensor(x_tensor):
        return 3 * x_tensor**3 - 2 * x_tensor + 1

    def f_float(x_val):
        return 3 * x_val**3 - 2 * x_val + 1
    
    x_val = 2.0
    x_tensor = create_tensors(x_val)

    deriv_numerical = numerical_derivative(f_float, x_val)
    deriv_automatic = automatic_derivative(f_tensor, x_tensor)
    
    # Reset gradient for the next test
    x_tensor.grad.zero_()

    assert math.isclose(deriv_numerical, deriv_automatic, rel_tol=TOLERANCE), \
        f"Mismatch for polynomial: Numerical={deriv_numerical}, Automatic={deriv_automatic}"

# Test case 4: A more complex function, e.g., f(x) = log(x^2 + 1)
def test_derivative_of_log_function_at_4():
    """
    Tests the derivative of f(x) = log(x^2 + 1) at x = 4.
    """
    def f_tensor(x_tensor):
        return torch.log(x_tensor**2 + 1)

    def f_float(x_val):
        return math.log(x_val**2 + 1)
    
    x_val = 4.0
    x_tensor = create_tensors(x_val)
    
    deriv_numerical = numerical_derivative(f_float, x_val)
    deriv_automatic = automatic_derivative(f_tensor, x_tensor)
    
    # Reset gradient for the next test
    x_tensor.grad.zero_()

    assert math.isclose(deriv_numerical, deriv_automatic, rel_tol=TOLERANCE), \
        f"Mismatch for log function: Numerical={deriv_numerical}, Automatic={deriv_automatic}"