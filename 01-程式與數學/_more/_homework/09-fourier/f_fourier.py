import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
import math

# 定義一個高斯函數
def gaussian(t, mu=0, sigma=1):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((t - mu) / sigma)**2)

def f(t):
    # return gaussian(t, mu=0, sigma=1)
    return np.sin(5*t)+np.cos(3*t)

# 時間域數據
t_values = np.linspace(-10, 10, 1000, endpoint=False) # 離散化時間
# 計算傅立葉轉換 (使用 FFT - 快速傅立葉轉換，是傅立葉轉換的離散版

# f_t = gaussian(t_values, mu=0, sigma=1)
f_t = f(t_values)

# 注意：FFT 處理的是離散數據，所以需要一些處理來得到正確的頻率軸
N = len(t_values)
T_sampling = t_values[1] - t_values[0] # 取樣間隔

F_omega = fft(f_t)
omega_values = fftfreq(N, T_sampling) * 2 * np.pi # 轉換到角頻率

# 為了繪圖方便，將頻率軸重新排序（讓零頻率在中間）
F_omega_shifted = np.fft.fftshift(F_omega)
omega_values_shifted = np.fft.fftshift(omega_values)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(t_values, f_t)
plt.title("f Function in Time Domain")
plt.xlabel("t")
plt.ylabel("f(t)")
plt.grid(True)

plt.subplot(1, 2, 2)
# 繪製幅度譜 (magnitude spectrum)
plt.plot(omega_values_shifted, np.abs(F_omega_shifted))
plt.title("Fourier Transform (Magnitude) of f Function")
plt.xlabel("Frequency (omega)")
plt.ylabel("|F(omega)|")
plt.grid(True)
plt.xlim(-10, 10) # 限制頻率顯示範圍以便觀察
plt.show()