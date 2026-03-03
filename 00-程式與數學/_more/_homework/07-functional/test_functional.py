import math

# 積分泛函的簡單實現（使用數值積分近似）
def integral_functional(func, lower_bound, upper_bound, num_steps=1000):
    """
    一個簡單的數值積分泛函，接收一個函數 func，
    並計算其在指定區間的近似定積分。
    """
    step_size = (upper_bound - lower_bound) / num_steps
    total_area = 0
    for i in range(num_steps):
        x = lower_bound + i * step_size
        total_area += func(x) * step_size
    return total_area

# 定義一些函數作為輸入
def linear_function(x):
    return x

def quadratic_function(x):
    return x ** 2

def sine_function(x):
    return math.sin(x)

# 使用泛函
integral_of_linear = integral_functional(linear_function, 0, 1)
integral_of_quadratic = integral_functional(quadratic_function, 0, 1)
integral_of_sine = integral_functional(sine_function, 0, math.pi) # sin(x) 從 0 到 pi 的積分應該是 2

print(f"Integral of x from 0 to 1: {integral_of_linear}")
print(f"Integral of x^2 from 0 to 1: {integral_of_quadratic}")
print(f"Integral of sin(x) from 0 to pi: {integral_of_sine}")


# 驗證我們的積分泛函
import pytest

# 使用 pytest 驗證
def test_integral_linear():
    # 積分 x 從 0 到 1 應該是 0.5
    assert abs(integral_functional(linear_function, 0, 1) - 0.5) < 0.01

def test_integral_quadratic():
    # 積分 x^2 從 0 到 1 應該是 1/3
    assert abs(integral_functional(quadratic_function, 0, 1) - (1/3)) < 0.01

def test_integral_sine():
    # 積分 sin(x) 從 0 到 pi 應該是 2
    assert abs(integral_functional(sine_function, 0, math.pi) - 2) < 0.01

# 你可以在終端機執行 `pytest your_file_name.py` 來運行這些測試
# 如果你想看到測試結果，需要先安裝 pytest: pip install pytest