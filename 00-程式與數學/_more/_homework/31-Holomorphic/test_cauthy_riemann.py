import pytest
from sympy import symbols, I, diff, re, im
import sympy
import cmath # Python's built-in complex number module

# 定義符號變數
x_sym, y_sym = symbols('x y')

# --- 數學函數定義 ---
def f_z_squared(z_val):
    """計算 f(z) = z^2 的值."""
    return z_val**2

def f_z_conjugate(z_val):
    """計算 f(z) = conj(z) 的值."""
    return z_val.conjugate()

# --- 幫助函數：提取 u 和 v ---
def get_uv_functions(f_z_expression_sym):
    """
    從符號表達式 f(z) 提取實部 u(x,y) 和虛部 v(x,y) 的符號函數。
    """
    u_sym = re(f_z_expression_sym)
    v_sym = im(f_z_expression_sym)
    return sympy.Lambda((x_sym, y_sym), u_sym), sympy.Lambda((x_sym, y_sym), v_sym)


# --- 證明中的關鍵步驟函數 ---
def calculate_f_prime_path1(f_func, u_func, v_func, z0, h=1e-6):
    """
    沿著實軸路徑計算 f'(z0) 的近似值 (x 趨近 x0)。
    f_func: 實際的複數函數，例如 f_z_squared
    u_func, v_func: 實部和虛部的符號函數，用於計算偏導數
    z0: 測試點 (complex)
    h: 微小增量
    """
    x0, y0 = z0.real, z0.imag

    # 用差商近似偏導數
    du_dx_approx = (u_func(x0 + h, y0) - u_func(x0, y0)) / h
    dv_dx_approx = (v_func(x0 + h, y0) - v_func(x0, y0)) / h

    return complex(du_dx_approx, dv_dx_approx)

def calculate_f_prime_path2(f_func, u_func, v_func, z0, h=1e-6):
    """
    沿著虛軸路徑計算 f'(z0) 的近似值 (y 趨近 y0)。
    f_func: 實際的複數函數
    u_func, v_func: 實部和虛部的符號函數
    z0: 測試點 (complex)
    h: 微小增量
    """
    x0, y0 = z0.real, z0.imag

    # 用差商近似偏導數
    dv_dy_approx = (v_func(x0, y0 + h) - v_func(x0, y0)) / h
    du_dy_approx = (u_func(x0, y0 + h) - u_func(x0, y0)) / h

    return complex(dv_dy_approx, -du_dy_approx) # 注意這裡的負號和虛部


# --- pytest 測試案例 ---

@pytest.fixture
def tolerance():
    """定義浮點數比較容許誤差."""
    return 1e-5

def test_cauchy_riemann_for_z_squared(tolerance):
    """
    驗證 f(z) = z^2 沿兩條路徑計算的 f'(z0) 是否相等。
    """
    print("\n--- 測試 f(z) = z^2 的柯西-黎曼方程式 ---")
    z_test = complex(1.0, 2.0) # 選擇一個測試點

    # 符號表達式用於提取 u, v
    z_sym = x_sym + I*y_sym
    f_sym = z_sym**2
    u_sym_func, v_sym_func = get_uv_functions(f_sym)

    f_prime_path1 = calculate_f_prime_path1(f_z_squared, u_sym_func, v_sym_func, z_test)
    f_prime_path2 = calculate_f_prime_path2(f_z_squared, u_sym_func, v_sym_func, z_test)

    print(f"對於 z = {z_test}:")
    print(f"f'(z) 沿實軸路徑近似: {f_prime_path1}")
    print(f"f'(z) 沿虛軸路徑近似: {f_prime_path2}")

    # 檢查實部和虛部是否足夠接近
    assert abs(f_prime_path1.real - f_prime_path2.real) < tolerance, "實部不相等"
    assert abs(f_prime_path1.imag - f_prime_path2.imag) < tolerance, "虛部不相等"

    # 我們也可以直接計算 f'(z) = 2z 來比較
    actual_f_prime = 2 * z_test
    print(f"實際 f'(z) = 2z: {actual_f_prime}")
    assert abs(f_prime_path1.real - actual_f_prime.real) < tolerance
    assert abs(f_prime_path1.imag - actual_f_prime.imag) < tolerance
    assert abs(f_prime_path2.real - actual_f_prime.real) < tolerance
    assert abs(f_prime_path2.imag - actual_f_prime.imag) < tolerance

    print("f(z) = z^2 滿足柯西-黎曼方程式的要求 (沿兩條路徑的導數相等)。")


def test_cauchy_riemann_for_z_conjugate_fails(tolerance):
    """
    驗證 f(z) = conj(z) 沿兩條路徑計算的 f'(z0) 是否不相等。
    """
    print("\n--- 測試 f(z) = conj(z) 的柯西-黎曼方程式 (預期失敗) ---")
    z_test = complex(1.0, 2.0)

    z_sym = x_sym + I*y_sym
    f_sym = sympy.conjugate(z_sym) # conj(z) = x - iy
    u_sym_func, v_sym_func = get_uv_functions(f_sym)

    f_prime_path1 = calculate_f_prime_path1(f_z_conjugate, u_sym_func, v_sym_func, z_test)
    f_prime_path2 = calculate_f_prime_path2(f_z_conjugate, u_sym_func, v_sym_func, z_test)

    print(f"對於 z = {z_test}:")
    print(f"f'(z) 沿實軸路徑近似: {f_prime_path1}")
    print(f"f'(z) 沿虛軸路徑近似: {f_prime_path2}")

    # 預期它們不相等，所以我們檢查它們是否確實不相等
    # 因為浮點數誤差，不能直接用 !=，而是檢查差異是否大於容許誤差
    assert abs(f_prime_path1.real - f_prime_path2.real) > tolerance or \
           abs(f_prime_path1.imag - f_prime_path2.imag) > tolerance, \
           "f(z) = conj(z) 的兩個路徑導數不應該相等，但它們卻相等了！"
    print("f(z) = conj(z) 不滿足柯西-黎曼方程式的要求 (沿兩條路徑的導數不相等，符合預期)。")