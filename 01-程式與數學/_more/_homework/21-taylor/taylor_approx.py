import pytest
import math
import numpy as np
import matplotlib.pyplot as plt
import sympy # 為了更精確地處理導數

# 1. 定義一個函數來計算階乘
def factorial(n):
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    if n == 0:
        return 1
    res = 1
    for i in range(1, n + 1):
        res *= i
    return res

# 我們定義要逼近的函數 f(x) = e^x
# 以及它的 n 階導數在 a=0 處的值
# 對於 f(x) = e^x，f^(n)(0) 總是 1

def taylor_series_e_x(x_val, num_terms):
    """
    計算 e^x 在 x=0 展開的泰勒多項式前 num_terms 項的和。
    """
    taylor_approx = 0
    for n in range(num_terms):
        # 對於 e^x，f^(n)(0) 總是 1
        term = (1 / factorial(n)) * (x_val**n)
        taylor_approx += term
    return taylor_approx

# 示範計算
x_to_approx = 0.5
print(f"真值 e^({x_to_approx}) = {math.exp(x_to_approx)}")

num_terms_list = [1, 2, 3, 4, 5, 10]
print("\n泰勒展開式逼近 e^x (展開點 a=0):")
for num_terms in num_terms_list:
    approx_val = taylor_series_e_x(x_to_approx, num_terms)
    print(f"使用 {num_terms} 項的近似值: {approx_val:.6f}, 誤差: {abs(math.exp(x_to_approx) - approx_val):.6f}")

# pytest 驗證
def test_taylor_series_e_x_accuracy():
    # 測試 e^0 = 1
    assert math.isclose(taylor_series_e_x(0, 5), 1.0, rel_tol=1e-9)

    # 測試 e^1 接近 2.71828
    assert math.isclose(taylor_series_e_x(1, 10), math.exp(1), rel_tol=1e-4)
    assert math.isclose(taylor_series_e_x(1, 15), math.exp(1), rel_tol=1e-9) # 項數越多越精確

    # 測試 e^-1 接近 0.367879
    assert math.isclose(taylor_series_e_x(-1, 10), math.exp(-1), rel_tol=1e-4)

# 2. 泛用泰勒展開函數 (允許自定義函數和展開點)
# 這需要我們能計算函數的 n 階導數

def get_nth_derivative(func_expr, var_sym, n, at_point):
    """
    使用 sympy 計算函數在某一點的 n 階導數值。
    func_expr: sympy 表達式
    var_sym: sympy 符號變量 (例如 x)
    n: 階數
    at_point: 計算導數值的點
    """
    if n == 0:
        return func_expr.subs(var_sym, at_point)
    
    derived_expr = func_expr
    for _ in range(n):
        derived_expr = sympy.diff(derived_expr, var_sym)
    
    return derived_expr.subs(var_sym, at_point)

def taylor_series(func_expr, var_sym, a, x_val, num_terms):
    """
    計算函數 func_expr 在點 a 展開的泰勒多項式前 num_terms 項的和。
    func_expr: sympy 表達式
    var_sym: sympy 符號變量
    a: 展開點
    x_val: 要計算近似值的點
    num_terms: 多項式的項數 (從 n=0 到 num_terms-1)
    """
    taylor_approx = 0
    for n in range(num_terms):
        f_nth_deriv_at_a = get_nth_derivative(func_expr, var_sym, n, a)
        term = (f_nth_deriv_at_a / factorial(n)) * ((x_val - a)**n)
        taylor_approx += term
    return taylor_approx

# 示範泛用泰勒展開
x_sym = sympy.Symbol('x')
f_x_sym = sympy.sin(x_sym) # 我們要逼近的函數是 sin(x)
a_point = 0 # 展開點為 0 (麥克勞林展開)
x_to_approx_gen = math.pi / 2 # 我們知道 sin(pi/2) = 1

print(f"\n真值 sin({x_to_approx_gen:.2f}) = {math.sin(x_to_approx_gen)}")

print("\n泰勒展開式逼近 sin(x) (展開點 a=0):")
for num_terms in num_terms_list:
    approx_val = taylor_series(f_x_sym, x_sym, a_point, x_to_approx_gen, num_terms)
    print(f"使用 {num_terms} 項的近似值: {approx_val:.6f}, 誤差: {abs(math.sin(x_to_approx_gen) - approx_val):.6f}")

# pytest 驗證泛用泰勒展開
def test_taylor_series_sin_x_accuracy():
    x_sym = sympy.Symbol('x')
    f_x_sym = sympy.sin(x_sym)
    
    # 測試 sin(0) = 0
    assert math.isclose(taylor_series(f_x_sym, x_sym, 0, 0, 5), 0.0, rel_tol=1e-9)

    # 測試 sin(pi/2) = 1
    # sin(x) 的泰勒展開在 x=pi/2 附近需要更多項才能收斂，因為 x 離展開點 a=0 較遠
    assert math.isclose(taylor_series(f_x_sym, x_sym, 0, math.pi / 2, 5), math.sin(math.pi / 2), rel_tol=1e-2)
    assert math.isclose(taylor_series(f_x_sym, x_sym, 0, math.pi / 2, 10), math.sin(math.pi / 2), rel_tol=1e-4)
    assert math.isclose(taylor_series(f_x_sym, x_sym, 0, math.pi / 2, 15), math.sin(math.pi / 2), rel_tol=1e-6)

    # 測試 sin(pi) = 0
    assert math.isclose(taylor_series(f_x_sym, x_sym, 0, math.pi, 15), math.sin(math.pi), rel_tol=1e-4) # 這裡會需要更多項，誤差會相對大一點


# 3. 圖形化展示
def plot_taylor_approximation(func_true, func_approx, x_range, title="Taylor Series Approximation"):
    """
    繪製函數真值與泰勒多項式近似值的圖。
    func_true: 實際函數 (接受 numpy 數組)
    func_approx: 泰勒多項式近似函數 (接受 numpy 數組)
    x_range: x 值的範圍 (例如 [-5, 5])
    """
    x_vals = np.linspace(x_range[0], x_range[1], 400)
    y_true = func_true(x_vals)
    y_approx = func_approx(x_vals)

    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_true, label='True Function', color='blue', linewidth=2)
    plt.plot(x_vals, y_approx, label='Taylor Approximation', color='red', linestyle='--', linewidth=2)
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True)
    plt.legend()
    plt.axvline(0, color='gray', linestyle=':', linewidth=0.8) # 顯示展開點
    plt.axhline(0, color='gray', linestyle=':', linewidth=0.8)
    plt.show()

# 繪製 e^x 的近似
num_terms_to_plot = 5
f_e_x_true = np.exp # 實際函數
f_e_x_approx_plot = lambda x: taylor_series_e_x(x, num_terms_to_plot)
plot_taylor_approximation(f_e_x_true, f_e_x_approx_plot, [-3, 3], 
                          f"Taylor Approximation of e^x ({num_terms_to_plot} terms at a=0)")

# 繪製 sin(x) 的近似
num_terms_to_plot_sin = 7 # sin(x) 的泰勒展開只有奇數項，所以 7 項實際上是 4 個非零項
f_sin_x_true = np.sin # 實際函數
# 將 sympy 表達式轉為 lambda 函數，以便繪圖
f_sin_x_approx_plot = lambda x: np.array([taylor_series(sympy.sin(x_sym), x_sym, 0, val, num_terms_to_plot_sin) for val in x])

plot_taylor_approximation(f_sin_x_true, f_sin_x_approx_plot, [-2*math.pi, 2*math.pi], 
                          f"Taylor Approximation of sin(x) ({num_terms_to_plot_sin} terms at a=0)")