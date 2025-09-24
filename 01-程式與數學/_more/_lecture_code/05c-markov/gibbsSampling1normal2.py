import numpy as np
import matplotlib.pyplot as plt



def gibbs_sampling(mu_x, mu_y, sigma_x, sigma_y, rho, n_samples, n_burnin):
    """
    使用 Gibbs Sampling 從二元常態分佈中抽取樣本。

    參數:
    mu_x (float): x 的平均值。
    mu_y (float): y 的平均值。
    sigma_x (float): x 的標準差。
    sigma_y (float): y 的標準差。
    rho (float): x 和 y 的相關係數。
    n_samples (int): 想要抽取的樣本總數。
    n_burnin (int): 捨棄的前期樣本數 (用來達到穩定狀態)。

    回傳:
    samples (numpy.ndarray): 包含 (x, y) 樣本的陣列。
    """
    # 初始化一個隨機的起始點
    x, y = 0.0, 0.0
    samples = []

    # 執行 Gibbs Sampling 迴圈
    for i in range(n_samples + n_burnin):
        # 根據 P(x|y) 的條件分佈更新 x
        # 條件分佈的平均值
        mean_x_cond = mu_x + rho * (sigma_x / sigma_y) * (y - mu_y)
        # 條件分佈的標準差
        std_x_cond = sigma_x * np.sqrt(1 - rho**2)
        x = np.random.normal(mean_x_cond, std_x_cond)
        
        # 根據 P(y|x) 的條件分佈更新 y
        # 條件分佈的平均值
        mean_y_cond = mu_y + rho * (sigma_y / sigma_x) * (x - mu_x)
        # 條件分佈的標準差
        std_y_cond = sigma_y * np.sqrt(1 - rho**2)
        y = np.random.normal(mean_y_cond, std_y_cond)

        # 捨棄 burn-in 階段的樣本
        if i >= n_burnin:
            samples.append([x, y])

    return np.array(samples)

# --- 參數設定 ---
mu_x_val, mu_y_val = 2, 5
sigma_x_val, sigma_y_val = 1.5, 2
rho_val = 0.8
n_samples_val = 10000
n_burnin_val = 1000

# 執行 Gibbs Sampling
samples = gibbs_sampling(mu_x_val, mu_y_val, sigma_x_val, sigma_y_val, rho_val, n_samples_val, n_burnin_val)

# --- 結果可視化 ---
plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(8, 6))
plt.scatter(samples[:, 0], samples[:, 1], s=5, alpha=0.3, label='Gibbs Sampling 樣本')
plt.title('Gibbs Sampling 從二元常態分佈中抽樣')
plt.xlabel('x')
plt.ylabel('y')
plt.axvline(mu_x_val, color='r', linestyle='--', label=f'x 平均值: {mu_x_val}')
plt.axhline(mu_y_val, color='g', linestyle='--', label=f'y 平均值: {mu_y_val}')
plt.legend()
plt.axis('equal')
plt.show()

# 驗證樣本的統計特性
print(f"樣本 x 的平均值: {np.mean(samples[:, 0]):.2f}, 期望值: {mu_x_val}")
print(f"樣本 y 的平均值: {np.mean(samples[:, 1]):.2f}, 期望值: {mu_y_val}")
print(f"樣本 x 的標準差: {np.std(samples[:, 0]):.2f}, 期望值: {sigma_x_val}")
print(f"樣本 y 的標準差: {np.std(samples[:, 1]):.2f}, 期望值: {sigma_y_val}")
print(f"樣本的相關係數: {np.corrcoef(samples.T)[0, 1]:.2f}, 期望值: {rho_val}")