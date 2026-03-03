import pytest
import sympy
import math

# 使用 sympy 進行符號微分，這樣我們可以得到精確的導數函數
x = sympy.Symbol('x')

def f_sym(x_sym):
    return x_sym**3 - 6*x_sym**2 + 5*x_sym + 1

# 計算 f(x) 的符號導數
f_prime_sym = sympy.diff(f_sym(x), x)
print(f"符號導數 f'(x): {f_prime_sym}")

# 將 sympy 表達式轉換為 Python 函數以便數值計算
f = sympy.lambdify(x, f_sym(x), 'numpy')
f_prime = sympy.lambdify(x, f_prime_sym, 'numpy')

def mean_value_theorem_verifier(func, func_prime, a, b, tolerance=0.02):
    """
    驗證平均值定理：尋找區間 [a, b] 內是否存在 c，使得 func_prime(c) = (func(b) - func(a)) / (b - a)。
    由於是數值方法，我們會在區間內取多個點進行檢查。
    """
    if not (b > a):
        raise ValueError("b 必須大於 a")

    # 計算割線的斜率 (Secant Line Slope)
    secant_slope = (func(b) - func(a)) / (b - a)

    print(f"區間 [{a}, {b}] 的割線斜率: {secant_slope:.5f}")

    # 在 [a, b] 區間內取多個點，檢查導數是否接近割線斜率
    # 我們不能保證找到「精確」的 c，但可以檢查是否存在一個點足夠接近。
    num_points = 1000
    points = [a + i * (b - a) / num_points for i in range(1, num_points)] # 避免取到端點 a, b

    found_c = False
    for c_candidate in points:
        derivative_at_c = func_prime(c_candidate)
        print(f"c_candidate={c_candidate:.3f}, f'(c_candidate)={derivative_at_c:.5f}") # 方便調試
        if math.isclose(derivative_at_c, secant_slope, rel_tol=tolerance, abs_tol=tolerance):
            print(f"在 c = {c_candidate:.5f} 處找到一個點，其導數 {derivative_at_c:.5f} 接近割線斜率。")
            found_c = True
            break # 找到一個就夠了，因為定理保證「至少存在一點」

    return found_c

# pytest 測試
def test_mean_value_theorem():
    a_val = 0
    b_val = 4
    # 期望找到 c，所以測試結果應該是 True
    assert mean_value_theorem_verifier(f, f_prime, a_val, b_val) == True

    a_val_neg = -1
    b_val_neg = 3
    assert mean_value_theorem_verifier(f, f_prime, a_val_neg, b_val_neg) == True

    # 也可以測試一個無法滿足 MVT 條件的函數，例如在區間內不連續或不可微
    # 但為了簡化，這裡只測試滿足條件的