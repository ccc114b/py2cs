import sympy
from sympy import symbols, diff

# Define the variable and the function
x = symbols('x')
f_x = x**5

# Initialize the function for the loop
current_f = f_x
x_val = 2

# Print the original function
print(f"原函數 f(x) = {f_x}")
print("-" * 30)

# Use a loop to perform continuous differentiation
for i in range(1, 10):
    current_f = diff(current_f, x)
    
    # Print the symbolic form
    print(f"{i}次微分 f{i}(x) = {current_f}")
    
    # Print the result after substituting x=2
    result_at_2 = current_f.subs(x, x_val)
    print(f"將 x={x_val} 代入結果為：{result_at_2}\n")
