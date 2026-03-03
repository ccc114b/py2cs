import numpy as np
import matplotlib.pyplot as plt
import math

# 定義一個方波函數
def square_wave(x, period=2*math.pi):
    # 假設週期是 2*pi，從 -pi 到 pi
    # 這裡我們讓方波在 [-pi, 0) 是 -1，在 [0, pi) 是 1
    # 為了處理週期性，先將 x 映射到一個週期內
    x_mod = (x + period / 2) % period - period / 2
    if x_mod >= 0:
        return 1
    else:
        return -1

# 計算傅立葉係數 (a_n 和 b_n)
# 對於方波 f(x) = 1 (0 < x < pi) 和 -1 (-pi < x < 0)
# a_0 = 0
# a_n = 0
# b_n = (4 / (n * pi)) 如果 n 是奇數, 0 如果 n 是偶數

def fourier_series_square_wave(x, num_terms=100, period=2*math.pi):
    sum_val = 0
    # a_0 對於方波是 0
    for n in range(1, num_terms + 1):
        # 對於這個方波，只有 b_n 存在且只在 n 為奇數時非零
        if n % 2 != 0: # n is odd
            b_n = (4 / (n * math.pi))
            sum_val += b_n * math.sin((2 * math.pi * n * x) / period)
    return sum_val

# 繪製結果
x_values = np.linspace(-3 * math.pi, 3 * math.pi, 500) # 繪製多個週期

# 原始方波
y_square_wave = [square_wave(x) for x in x_values]

# 傅立葉級數近似
num_terms_to_plot = [1, 3, 5, 20] # 嘗試不同數量的項來觀察近似效果

plt.figure(figsize=(12, 8))
plt.plot(x_values, y_square_wave, label="Original Square Wave", linestyle='--', color='gray', linewidth=2)

for num_terms in num_terms_to_plot:
    y_fourier_approx = [fourier_series_square_wave(x, num_terms=num_terms) for x in x_values]
    plt.plot(x_values, y_fourier_approx, label=f"Fourier Series ({num_terms} terms)")

plt.title("Fourier Series Approximation of a Square Wave")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)
plt.legend()
plt.ylim(-1.5, 1.5)
plt.axvline(x=0, color='r', linestyle=':', linewidth=0.8)
plt.axvline(x=math.pi, color='r', linestyle=':', linewidth=0.8)
plt.axvline(x=-math.pi, color='r', linestyle=':', linewidth=0.8)
plt.show()