"""
pytest tests for the calculus.py module.
"""

import math
import pytest
from calculus import numerical_derivative, numerical_integral

# A small tolerance for floating-point comparisons
TOLERANCE = 1e-6

# --- Tests for numerical_derivative ---

def test_derivative_of_x_squared():
    """
    Tests the derivative of f(x) = x^2.
    The exact derivative is f'(x) = 2x.
    """
    def f(x):
        return x**2
    
    x = 3.0
    expected = 2 * x
    assert abs(numerical_derivative(f, x) - expected) < TOLERANCE

def test_derivative_of_sin_x_at_pi_over_2():
    """
    Tests the derivative of f(x) = sin(x) at x = pi/2.
    The exact derivative is f'(x) = cos(x).
    cos(pi/2) = 0.
    """
    def f(x):
        return math.sin(x)
    
    x = math.pi / 2
    expected = math.cos(x)
    assert abs(numerical_derivative(f, x) - expected) < TOLERANCE

def test_derivative_of_constant_function():
    """
    Tests the derivative of a constant function, which should be 0.
    """
    def f(x):
        return 5.0
    
    x = 10.0
    expected = 0.0
    assert abs(numerical_derivative(f, x) - expected) < TOLERANCE

# --- Tests for numerical_integral ---

def test_integral_of_x_squared_from_0_to_2():
    """
    Tests the integral of f(x) = x^2 from 0 to 2.
    The exact integral is [x^3/3] from 0 to 2, which is 8/3.
    """
    def f(x):
        return x**2
    
    a, b = 0, 2
    expected = 8 / 3
    assert abs(numerical_integral(f, a, b) - expected) < 1e-4 # Use a larger tolerance for integration

def test_integral_of_cos_x_from_0_to_pi():
    """
    Tests the integral of f(x) = cos(x) from 0 to pi.
    The exact integral is [sin(x)] from 0 to pi, which is sin(pi) - sin(0) = 0 - 0 = 0.
    """
    def f(x):
        return math.cos(x)
    
    a, b = 0, math.pi
    expected = 0.0
    assert abs(numerical_integral(f, a, b) - expected) < 1e-4

def test_integral_with_swapped_bounds():
    """
    Tests that the integral works correctly if the bounds are swapped.
    The integral from 2 to 0 should be the negative of the integral from 0 to 2.
    """
    def f(x):
        return x**2
    
    a, b = 2, 0
    expected_positive = 8 / 3
    assert abs(numerical_integral(f, a, b) - (-expected_positive)) < 1e-4
