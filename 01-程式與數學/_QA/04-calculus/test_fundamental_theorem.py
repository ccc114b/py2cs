"""
pytest test to verify the Fundamental Theorem of Calculus numerically.
"""

import math
import pytest
from calculus import numerical_derivative, numerical_integral

# A small tolerance for floating-point comparisons
TOLERANCE = 1e-4

def test_fundamental_theorem_of_calculus():
    """
    Verifies the Fundamental Theorem of Calculus (part 1) numerically.
    It states that if F(x) = integral_a^x f(t) dt, then F'(x) = f(x).
    
    We will:
    1. Define an original function f(x).
    2. Create a new function F(x) that calculates the numerical integral of f(t) from 0 to x.
    3. Compute the numerical derivative of F(x) at a test point.
    4. Assert that the result of the derivative is very close to the original function f(x)
       at that same point.
    """
    
    # Step 1: Define the original function f(x)
    # Let's use a simple and smooth function, like f(x) = x^2 + 2x
    def f(x):
        return x**2 + 2 * x

    # Step 2: Create the "antiderivative" function F(x) using numerical integration
    # F(x) = integral of f(t) from 0 to x
    def F(x):
        # The integral starts from 0, as a reference point.
        # We can choose any starting point for the integral.
        return numerical_integral(f, 0, x)
    
    # Step 3: Test the theorem at several points
    test_points = [1.0, 2.5, -3.0, 5.0]

    for x in test_points:
        # Compute the numerical derivative of F(x) at the point x
        derived_result = numerical_derivative(F, x)
        
        # Compute the expected value from the original function f(x)
        expected_result = f(x)

        # Step 4: Assert that the derivative of the integral is close to the original function
        assert abs(derived_result - expected_result) < TOLERANCE, \
            f"Test failed for x={x}. Derived: {derived_result:.6f}, Expected: {expected_result:.6f}"

# You can add another test with a different function, e.g., trigonometric
def test_fundamental_theorem_with_sin():
    """
    Tests the fundamental theorem using the function f(x) = cos(x).
    """
    def f(x):
        return math.cos(x)

    def F(x):
        return numerical_integral(f, 0, x)

    x = math.pi / 4
    derived_result = numerical_derivative(F, x)
    expected_result = f(x)

    assert abs(derived_result - expected_result) < TOLERANCE, \
        f"Test failed for x={x}. Derived: {derived_result:.6f}, Expected: {expected_result:.6f}"
