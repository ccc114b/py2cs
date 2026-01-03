import mpmath
import matplotlib.pyplot as plt
import numpy as np

# 設定 mpmath 的浮點數精度
mpmath.dps = 25 # 設定 25 位有效數字的精度

def plot_zeta_real_part(start_t, end_t, num_points):
    """
    繪製黎曼zeta函數在臨界線 Re(s) = 0.5 上實部的變化。
    零點發生在實部和虛部同時為零的地方。
    這個函數只是展示 Re(zeta(0.5 + it))。
    """
    t_values = np.linspace(start_t, end_t, num_points)
    real_parts = []
    imag_parts = []

    for t in t_values:
        s = mpmath.mpc(0.5, t) # s = 0.5 + it
        zeta_s = mpmath.zeta(s)
        real_parts.append(zeta_s.real)
        imag_parts.append(zeta_s.imag)

    plt.figure(figsize=(12, 6))
    plt.plot(t_values, real_parts, label='Real part of zeta(0.5 + it)', color='blue')
    plt.axhline(0, color='red', linestyle='--', linewidth=0.8, label='Zero line')
    plt.title('Real Part of Riemann Zeta Function on the Critical Line (Re(s)=0.5)')
    plt.xlabel('t (Imaginary part of s)')
    plt.ylabel('Re(zeta(0.5 + it))')
    plt.grid(True)
    plt.legend()
    plt.show()

    # 為了更直觀，也可以試著找出變號的地方作為零點的初步猜測
    print("\n大致的零點位置 (實部變號處):")
    for i in range(1, len(real_parts)):
        if real_parts[i-1] * real_parts[i] < 0: # 如果有變號，表示可能穿越了零點
            t_approx = (t_values[i-1] + t_values[i]) / 2
            print(f"  在 t 約 {t_approx:.4f} 附近可能存在零點")

    # 注意：更精確的零點需要專門的數值方法，例如尋根演算法。
    # mpmath 庫內建了尋找 zeta 零點的功能，但這通常用於驗證，而不是「證明」。

# 繪製 zeta 函數在臨界線上實部的變化
# 已知第一個非平凡零點的虛部約為 14.1347
# 第二個約為 21.0220
# 第三個約為 25.0108
plot_zeta_real_part(0, 30, 1000)

"""
# 也可以直接用 mpmath 內建功能查找零點（這需要時間）
print("\n使用 mpmath 庫查找前幾個非平凡零點的虛部:")
# mpmath.zeta_zeros() 函數會返回指定數量（或範圍內）的 zeta 函數零點的虛部。
# 它們都是在 Re(s) = 0.5 的前提下找到的。
try:
    first_few_zeros_t = [z.imag for z in mpmath.find_zeta_zeros(0, 30)]
    print(f"前幾個非平凡零點的虛部: {first_few_zeros_t}")
except Exception as e:
    print(f"無法查找 zeta 零點: {e}")
    print("這可能是因為 mpmath 依賴於其他數值庫，或者在某些環境下無法順利執行。")
"""
