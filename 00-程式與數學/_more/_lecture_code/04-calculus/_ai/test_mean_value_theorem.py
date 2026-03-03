"""
pytest test to verify the Mean Value Theorem numerically.
"""

import pytest
from calculus import numerical_derivative

# A small tolerance for floating-point comparisons
TOLERANCE = 1e-4

# A small search step for finding the point c
SEARCH_STEP = 1e-3

def test_mean_value_theorem_for_parabola():
    """
    Verifies the Mean Value Theorem for the function f(x) = x^2 on the interval [1, 5].
    
    Expected result:
    - Average slope = (f(5) - f(1)) / (5 - 1) = (25 - 1) / 4 = 6
    - Derivative f'(x) = 2x
    - 2c = 6 -> c = 3.
    """
    def f(x):
        return x**2
    
    a, b = 1, 5
    
    # Calculate the average slope of the secant line
    average_slope = (f(b) - f(a)) / (b - a)
    
    # Search for a point c in the interval (a, b) where the tangent slope equals the average slope.
    found_c = False
    
    # Use a simple linear search for demonstration.
    # A more robust solution might use a binary search or optimization algorithm.
    x_test = a + SEARCH_STEP
    while x_test < b:
        # Calculate the numerical derivative at the current point
        tangent_slope = numerical_derivative(f, x_test)
        
        # Check if the tangent slope is close to the average slope
        if abs(tangent_slope - average_slope) < TOLERANCE:
            print(f"Found a point c = {x_test:.4f} where f'(c) ≈ {tangent_slope:.4f} "
                  f"which is close to average slope {average_slope:.4f}")
            found_c = True
            break
        
        x_test += SEARCH_STEP

    # Assert that we have found at least one such point c
    assert found_c, "Failed to find a point c satisfying the Mean Value Theorem."

def test_mean_value_theorem_for_cubic_function():
    """
    Verifies the Mean Value Theorem for the function f(x) = x^3 - x on the interval [-1, 2].
    
    Expected result:
    - Average slope = (f(2) - f(-1)) / (2 - (-1)) = (6 - 0) / 3 = 2
    - Derivative f'(x) = 3x^2 - 1
    - 3x^2 - 1 = 2 -> 3x^2 = 3 -> x^2 = 1 -> x = ±1.
    - We should find x = 1 in the interval (-1, 2).
    """
    def f(x):
        return x**3 - x
    
    a, b = -1, 2
    
    average_slope = (f(b) - f(a)) / (b - a)
    
    found_c = False
    x_test = a + SEARCH_STEP
    while x_test < b:
        tangent_slope = numerical_derivative(f, x_test)
        
        if abs(tangent_slope - average_slope) < TOLERANCE:
            print(f"Found a point c = {x_test:.4f} where f'(c) ≈ {tangent_slope:.4f} "
                  f"which is close to average slope {average_slope:.4f}")
            found_c = True
            break
            
        x_test += SEARCH_STEP

    assert found_c, "Failed to find a point c satisfying the Mean Value Theorem."