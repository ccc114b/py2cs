import pytest
import math

# 我們定義一個函數來近似計算極限
# 在實際數學中，極限是無限趨近的過程，但在程式中我們只能取一個足夠接近的值來代表。
def calculate_limit_approx(func, c, delta=1e-6):
    """
    近似計算 func 在 x 趨近於 c 時的極限。
    我們取 c 附近的一個點 (c + delta) 的函數值作為近似。
    """
    return func(c + delta)

# 定義兩個簡單的函數
def f(x):
    return x**2 + 1

def g(x):
    return 2 * x - 3

# 我們要驗證在 x 趨近於 c=2 時的極限定理
c_val = 2

# 計算 f(x) 和 g(x) 在 x 趨近 2 時的理論極限值
L = f(c_val)  # 對於多項式函數，極限值就是直接代入
M = g(c_val)  # 對於多項式函數，極限值就是直接代入

print(f"理論上 lim(x->{c_val}) f(x) = {L}")
print(f"理論上 lim(x->{c_val}) g(x) = {M}")

# pytest 測試
def test_limit_sum_rule():
    # 驗證 lim(f(x) + g(x)) = L + M
    sum_func = lambda x: f(x) + g(x)
    assert math.isclose(calculate_limit_approx(sum_func, c_val), L + M, rel_tol=1e-5)

def test_limit_diff_rule():
    # 驗證 lim(f(x) - g(x)) = L - M
    diff_func = lambda x: f(x) - g(x)
    assert math.isclose(calculate_limit_approx(diff_func, c_val), L - M, rel_tol=1e-5)

def test_limit_product_rule():
    # 驗證 lim(f(x) * g(x)) = L * M
    prod_func = lambda x: f(x) * g(x)
    assert math.isclose(calculate_limit_approx(prod_func, c_val), L * M, rel_tol=1e-5)

def test_limit_constant_multiple_rule():
    # 驗證 lim(k * f(x)) = k * L
    k = 5
    const_mult_func = lambda x: k * f(x)
    assert math.isclose(calculate_limit_approx(const_mult_func, c_val), k * L, rel_tol=1e-5)

def test_limit_quotient_rule():
    # 驗證 lim(f(x) / g(x)) = L / M (M != 0)
    # 這裡確保 g(c_val) 不為 0
    if M != 0:
        quotient_func = lambda x: f(x) / g(x)
        assert math.isclose(calculate_limit_approx(quotient_func, c_val), L / M, rel_tol=1e-5)
    else:
        pytest.skip("Denominator M is zero, cannot test quotient rule.")