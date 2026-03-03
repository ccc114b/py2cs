"""
A simple module for numerical calculus operations.
"""

def numerical_derivative(f, x, h=1e-6):
    """
    Computes the numerical derivative of a function f at a point x
    using the symmetric difference quotient.

    Args:
        f (callable): The function to differentiate.
        x (float): The point at which to compute the derivative.
        h (float): The step size for the difference quotient.

    Returns:
        float: The approximate derivative of f at x.
    """
    return (f(x + h) - f(x - h)) / (2 * h)

def numerical_integral(f, a, b, n=1000):
    """
    Computes the numerical integral of a function f from a to b
    using the trapezoidal rule.

    Args:
        f (callable): The function to integrate.
        a (float): The lower bound of integration.
        b (float): The upper bound of integration.
        n (int): The number of trapezoids to use.

    Returns:
        float: The approximate integral of f from a to b.
    """
    if a > b:
        a, b = b, a
        sign = -1
    else:
        sign = 1

    h = (b - a) / n
    integral_sum = 0.5 * (f(a) + f(b))
    for i in range(1, n):
        integral_sum += f(a + i * h)
    
    return sign * h * integral_sum

if __name__ == '__main__':
    # Example usage:
    # --- Derivative ---
    print("--- Numerical Derivative ---")
    
    # Differentiate f(x) = x^2 at x = 3
    # Exact derivative is 2x, so at x=3 it should be 6
    def f_squared(x):
        return x**2
    
    deriv_at_3 = numerical_derivative(f_squared, 3)
    print(f"Derivative of x^2 at x=3 is approximately: {deriv_at_3:.6f}")

    # Differentiate f(x) = sin(x) at x = pi/2
    # Exact derivative is cos(x), so at x=pi/2 it should be 0
    import math
    def f_sin(x):
        return math.sin(x)

    deriv_sin_at_pi_over_2 = numerical_derivative(f_sin, math.pi / 2)
    print(f"Derivative of sin(x) at pi/2 is approximately: {deriv_sin_at_pi_over_2:.6f}")
    
    # --- Integral ---
    print("\n--- Numerical Integral ---")
    
    # Integrate f(x) = x^2 from 0 to 2
    # Exact integral is [x^3/3] from 0 to 2, which is 8/3 ≈ 2.666667
    integral_0_to_2 = numerical_integral(f_squared, 0, 2)
    print(f"Integral of x^2 from 0 to 2 is approximately: {integral_0_to_2:.6f}")

    # Integrate f(x) = cos(x) from 0 to pi
    # Exact integral is [sin(x)] from 0 to pi, which is 0
    def f_cos(x):
        return math.cos(x)

    integral_cos_0_to_pi = numerical_integral(f_cos, 0, math.pi)
    print(f"Integral of cos(x) from 0 to pi is approximately: {integral_cos_0_to_pi:.6f}")