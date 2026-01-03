import math
import numpy as np
import pytest

# 數值積分函數（梯形法則）
def numerical_integral(func, a, b, num_steps=10000):
    """計算函數 func 在區間 [a, b] 上的數值積分"""
    h = (b - a) / num_steps
    total = (func(a) + func(b)) / 2.0
    for i in range(1, num_steps):
        total += func(a + i * h)
    return total * h

# 函數的內積（在 [-pi, pi] 區間）
def inner_product(f, g, a=-math.pi, b=math.pi):
    """計算兩個實數值函數的內積"""
    # 這裡我們假設是實數值函數，所以沒有共軛
    integrand = lambda x: f(x) * g(x)
    return numerical_integral(integrand, a, b)

# 傅立葉基底函數
def const_one(x):
    return 1.0

def sin_nx(n):
    return lambda x: math.sin(n * x)

def cos_nx(n):
    return lambda x: math.cos(n * x)

# 測試用例
integration_interval = [-math.pi, math.pi]
tolerance = 1e-3 # 由於是數值積分，設定一個容忍度

def test_orthogonality_sin_cos():
    # sin(nx) 和 cos(mx) 應該正交
    n_values = [1, 2, 3]
    m_values = [1, 2, 3]
    for n in n_values:
        for m in m_values:
            # 任何 sin(nx) 和 cos(mx) 都正交
            assert abs(inner_product(sin_nx(n), cos_nx(m), *integration_interval)) < tolerance

def test_orthogonality_sin_different_n():
    # sin(nx) 和 sin(mx) 當 n != m 時應該正交
    n1, n2 = 1, 2
    assert abs(inner_product(sin_nx(n1), sin_nx(n2), *integration_interval)) < tolerance

def test_orthogonality_cos_different_n():
    # cos(nx) 和 cos(mx) 當 n != m 時應該正交
    n1, n2 = 1, 2
    assert abs(inner_product(cos_nx(n1), cos_nx(n2), *integration_interval)) < tolerance

def test_orthogonality_const_sin():
    # 1 和 sin(nx) 應該正交
    n = 1
    assert abs(inner_product(const_one, sin_nx(n), *integration_interval)) < tolerance

def test_orthogonality_const_cos():
    # 1 和 cos(nx) 應該正交
    n = 1
    assert abs(inner_product(const_one, cos_nx(n), *integration_interval)) < tolerance

def test_norm_sin_n():
    # ||sin(nx)||^2 = pi (對於 n >= 1)
    n = 1
    norm_squared = inner_product(sin_nx(n), sin_nx(n), *integration_interval)
    assert abs(norm_squared - math.pi) < tolerance

def test_norm_cos_n():
    # ||cos(nx)||^2 = pi (對於 n >= 1)
    n = 1
    norm_squared = inner_product(cos_nx(n), cos_nx(n), *integration_interval)
    assert abs(norm_squared - math.pi) < tolerance

def test_norm_const_one():
    # ||1||^2 = 2*pi
    norm_squared = inner_product(const_one, const_one, *integration_interval)
    assert abs(norm_squared - (2 * math.pi)) < tolerance