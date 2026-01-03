import pytest
import sympy
import math

# 使用 sympy 進行符號微分和積分
x = sympy.Symbol('x')

def f_sym_ftc(x_sym):
    return x_sym**2 + 2*x_sym + 1 # 我們選擇一個簡單的多項式函數 f(x)

# 計算 f(x) 的符號不定積分 (反導數 F(x))
F_sym = sympy.integrate(f_sym_ftc(x), x)
print(f"符號函數 f(x): {f_sym_ftc(x)}")
print(f"符號反導數 F(x): {F_sym}")

# 將 sympy 表達式轉換為 Python 函數以便數值計算
f_ftc = sympy.lambdify(x, f_sym_ftc(x), 'numpy')
F_ftc = sympy.lambdify(x, F_sym, 'numpy')

def riemann_sum_integral(func, a, b, num_rectangles=10000):
    """
    使用右黎曼和近似計算定積分。
    """
    if not (b > a):
        raise ValueError("b 必須大於 a")

    dx = (b - a) / num_rectangles
    integral_approx = 0
    for i in range(num_rectangles):
        x_val = a + (i + 1) * dx # 右端點
        integral_approx += func(x_val) * dx
    return integral_approx

# pytest 測試
def test_fundamental_theorem_of_calculus():
    a_val = 0
    b_val = 3

    # 1. 使用黎曼和近似計算定積分
    approx_integral = riemann_sum_integral(f_ftc, a_val, b_val)
    print(f"使用黎曼和計算的近似積分值: {approx_integral:.5f}")

    # 2. 使用反導數計算 F(b) - F(a)
    exact_integral_ftc = F_ftc(b_val) - F_ftc(a_val)
    print(f"使用微積分基本定理計算的精確積分值: {exact_integral_ftc:.5f}")

    # 驗證兩者是否足夠接近
    assert math.isclose(approx_integral, exact_integral_ftc, rel_tol=1e-3, abs_tol=1e-3) # rel_tol 可以稍微大一點因為黎曼和是近似

    # 另一個區間
    a_val_neg = -2
    b_val_neg = 1
    approx_integral_neg = riemann_sum_integral(f_ftc, a_val_neg, b_val_neg)
    exact_integral_ftc_neg = F_ftc(b_val_neg) - F_ftc(a_val_neg)
    assert math.isclose(approx_integral_neg, exact_integral_ftc_neg, rel_tol=1e-3, abs_tol=1e-3)