import sympy
from sympy import symbols, I, diff

# 定義符號變數 x, y
x, y = symbols('x y')

# 定義複數 z
z = x + I*y

# 函數 f(z) = z^2
f_z_squared = z**2

# 將 f(z) 展開成實部 u(x, y) 和虛部 v(x, y)
# f(z) = (x + iy)^2 = x^2 + 2ixy - y^2 = (x^2 - y^2) + i(2xy)
u_squared = (x**2 - y**2)
v_squared = (2*x*y)

print(f"對於 f(z) = z^2:")
print(f"u(x, y) = {u_squared}")
print(f"v(x, y) = {v_squared}")

# 計算偏導數
du_dx_squared = diff(u_squared, x)
du_dy_squared = diff(u_squared, y)
dv_dx_squared = diff(v_squared, x)
dv_dy_squared = diff(v_squared, y)

print(f"\n偏導數：")
print(f"du/dx = {du_dx_squared}")
print(f"du/dy = {du_dy_squared}")
print(f"dv/dx = {dv_dx_squared}")
print(f"dv/dy = {dv_dy_squared}")

# 驗證柯西-黎曼方程式
cr_eq1_satisfied = (du_dx_squared == dv_dy_squared)
cr_eq2_satisfied = (du_dy_squared == -dv_dx_squared)

print(f"\n柯西-黎曼方程式驗證：")
print(f"∂u/∂x == ∂v/∂y : {cr_eq1_satisfied}")
print(f"∂u/∂y == -∂v/∂x : {cr_eq2_satisfied}")

# 結論
if cr_eq1_satisfied and cr_eq2_satisfied:
    print("結論：f(z) = z^2 滿足柯西-黎曼方程式，因此是全純函數。")
else:
    print("結論：f(z) = z^2 不滿足柯西-黎曼方程式。")