"""
Module for numerical and automatic differentiation.
"""

import torch

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

def automatic_derivative(f, x_tensor):
    """
    Computes the automatic derivative of a function f at a point x
    using PyTorch's autograd.

    Args:
        f (callable): The function to differentiate. It must accept a PyTorch tensor.
        x_tensor (torch.Tensor): The point at which to compute the derivative,
                                 with requires_grad=True.

    Returns:
        float: The derivative of f at x, computed automatically.
    """
    # Compute the function value
    y_tensor = f(x_tensor)
    
    # Perform backpropagation to compute the gradient
    y_tensor.backward()
    
    # The gradient is stored in the .grad attribute of the input tensor
    return x_tensor.grad.item()

# Example usage (for demonstration)
if __name__ == '__main__':
    def f_squared(x):
        return x**2
    
    x_val = 3.0
    x_tensor = torch.tensor(x_val, requires_grad=True)

    # Numerical
    deriv_numerical = numerical_derivative(f_squared, x_val)
    print(f"Numerical derivative of x^2 at x={x_val}: {deriv_numerical}")
    
    # Automatic
    deriv_automatic = automatic_derivative(f_squared, x_tensor)
    print(f"Automatic derivative of x^2 at x={x_val}: {deriv_automatic}")