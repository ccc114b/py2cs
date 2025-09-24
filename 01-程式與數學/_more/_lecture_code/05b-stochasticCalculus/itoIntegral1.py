import numpy as np
import matplotlib.pyplot as plt

def ito_integral_simulation(T, N, num_paths=1):
    """
    使用蒙地卡羅方法模擬伊藤積分：I = ∫_0^T W_s dW_s

    參數:
    T (float): 模擬的總時間
    N (int): 時間步數
    num_paths (int): 模擬的路徑數

    回傳:
    tuple: (numerical_integrals, analytical_integrals)
    """
    dt = T / N  # 時間步長

    # 生成多條布朗運動路徑
    dW = np.sqrt(dt) * np.random.randn(N, num_paths)
    W = np.cumsum(dW, axis=0)
    
    # 為了計算方便，在路徑開始時增加一個 W_0 = 0 的點
    W = np.insert(W, 0, 0, axis=0)

    # 計算伊藤積分的數值解
    # 根據定義，我們使用 W_t_i 乘以 dW_i
    # W 的尺寸是 (N+1, num_paths)，dW 的尺寸是 (N, num_paths)
    # 我們需要 W_t_i，也就是除了最後一個點之外的 W
    numerical_integrals = np.sum(W[:-1, :] * dW, axis=0)

    # 計算伊藤引理的解析解
    # I_T = 1/2 * (W_T^2 - T)
    W_T = W[-1, :]  # 取布朗運動的最終值
    analytical_integrals = 0.5 * (W_T**2 - T)

    return numerical_integrals, analytical_integrals

# 參數設定
T = 1.0  # 總時間
N = 1000  # 時間步數
num_paths = 10000  # 模擬的路徑數

# 執行模擬
numerical_results, analytical_results = ito_integral_simulation(T, N, num_paths)

# 繪製直方圖以比較結果的分佈
plt.figure(figsize=(10, 6))
plt.hist(numerical_results, bins=50, alpha=0.5, label='Numerical Result', density=True)
plt.hist(analytical_results, bins=50, alpha=0.5, label='Analytical Result (Ito\'s Lemma)', density=True)
plt.title(f"Comparison of Numerical and Analytical Ito Integrals (T={T}, N={N})")
plt.xlabel("Integral Value")
plt.ylabel("Probability Density")
plt.legend()
plt.grid(True)
plt.show()

# 顯示平均值和標準差以進行定量比較
print(f"Number of paths: {num_paths}, Time steps: {N}")
print(f"Mean of Numerical Results: {np.mean(numerical_results):.4f}")
print(f"Mean of Analytical Results: {np.mean(analytical_results):.4f}")
print("-" * 30)
print(f"Std Dev of Numerical Results: {np.std(numerical_results):.4f}")
print(f"Std Dev of Analytical Results: {np.std(analytical_results):.4f}")
