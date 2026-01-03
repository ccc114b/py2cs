import pytest

# 定義一些函數
def f(x):
    return x**2

def g(x):
    return 2 * x + 1

def h(x):
    return x - 3

def z(x): # 零函數
    return 0

# 函數加法
def add_functions(func1, func2):
    return lambda x: func1(x) + func2(x)

# 純量乘法
def scalar_multiply_function(scalar, func):
    return lambda x: scalar * func(x)

# 負函數
def negative_function(func):
    return lambda x: -func(x)

# 測試點
test_points = [0, 1, -1, 0.5, 10]

# --- 函數加法相關性質的驗證 ---

def test_function_addition_closure():
    # 閉合性：兩個函數相加仍然是函數
    f_plus_g = add_functions(f, g)
    for x in test_points:
        assert f_plus_g(x) == f(x) + g(x)

def test_function_addition_commutative():
    # 交換律：f + g == g + f
    f_plus_g = add_functions(f, g)
    g_plus_f = add_functions(g, f)
    for x in test_points:
        assert f_plus_g(x) == g_plus_f(x)

def test_function_addition_associative():
    # 結合律：(f + g) + h == f + (g + h)
    f_plus_g_plus_h_left = add_functions(add_functions(f, g), h)
    f_plus_g_plus_h_right = add_functions(f, add_functions(g, h))
    for x in test_points:
        assert f_plus_g_plus_h_left(x) == f_plus_g_plus_h_right(x)

def test_function_addition_zero_element():
    # 零向量：f + z == f
    f_plus_z = add_functions(f, z)
    for x in test_points:
        assert f_plus_z(x) == f(x)

def test_function_addition_negative_element():
    # 負向量：f + (-f) == z
    f_plus_neg_f = add_functions(f, negative_function(f))
    for x in test_points:
        assert f_plus_neg_f(x) == z(x)

# --- 純量乘法相關性質的驗證 ---

def test_scalar_multiplication_closure():
    # 閉合性：純量乘以函數仍然是函數
    c = 3
    c_times_f = scalar_multiply_function(c, f)
    for x in test_points:
        assert c_times_f(x) == c * f(x)

def test_scalar_multiplication_associative():
    # 純量乘法結合律：c * (d * f) == (c * d) * f
    c, d = 2, 3
    c_d_f_left = scalar_multiply_function(c, scalar_multiply_function(d, f))
    c_d_f_right = scalar_multiply_function(c * d, f)
    for x in test_points:
        assert c_d_f_left(x) == c_d_f_right(x)

def test_scalar_multiplication_identity():
    # 純量乘法單位元：1 * f == f
    one_times_f = scalar_multiply_function(1, f)
    for x in test_points:
        assert one_times_f(x) == f(x)

def test_scalar_multiplication_distributive_over_function_addition():
    # 純量對函數加法的分配律：c * (f + g) == c * f + c * g
    c = 2
    c_times_f_plus_g_left = scalar_multiply_function(c, add_functions(f, g))
    c_times_f_plus_g_right = add_functions(scalar_multiply_function(c, f), scalar_multiply_function(c, g))
    for x in test_points:
        assert c_times_f_plus_g_left(x) == c_times_f_plus_g_right(x)

def test_scalar_multiplication_distributive_over_scalar_addition():
    # 純量對純量加法的分配律：(c + d) * f == c * f + d * f
    c, d = 2, 3
    c_plus_d_times_f_left = scalar_multiply_function(c + d, f)
    c_plus_d_times_f_right = add_functions(scalar_multiply_function(c, f), scalar_multiply_function(d, f))
    for x in test_points:
        assert c_plus_d_times_f_left(x) == c_plus_d_times_f_right(x)

