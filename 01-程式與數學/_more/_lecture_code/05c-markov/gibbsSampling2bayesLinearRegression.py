import numpy as np
import matplotlib.pyplot as plt

# --- 1. 生成模擬資料 ---
# 真正的參數值
true_beta_0 = 3
true_beta_1 = 1.5
true_sigma2 = 4

np.random.seed(42)
x = np.linspace(0, 10, 50)
y = true_beta_0 + true_beta_1 * x + np.random.normal(0, np.sqrt(true_sigma2), size=len(x))

# 將資料轉換成設計矩陣 X
X = np.vstack([np.ones_like(x), x]).T

# --- 2. 設定先驗參數 ---
# 簡單的先驗，假設 beta_0 和 beta_1 的平均值為0，方差為10
prior_mean = np.zeros(2)
prior_cov = np.eye(2) * 10
prior_cov_inv = np.linalg.inv(prior_cov)

# Inverse-Gamma 的先驗參數
prior_alpha = 1
prior_beta = 1

# --- 3. 執行 Gibbs Sampling ---
def gibbs_sampling_bayesian_regression(X, y, n_samples, n_burnin):
    # 初始化參數
    n_features = X.shape[1]
    beta = np.zeros(n_features)
    sigma2 = 1.0  # 初始值

    # 儲存樣本
    beta_samples = np.zeros((n_samples, n_features))
    sigma2_samples = np.zeros(n_samples)

    for i in range(n_samples + n_burnin):
        # a. 根據資料和目前的 sigma2 更新 beta
        # 條件後驗分佈的共變異矩陣 (covariance matrix)
        post_cov = np.linalg.inv(prior_cov_inv + (1/sigma2) * X.T @ X)
        
        # 條件後驗分佈的平均值 (mean)
        post_mean = post_cov @ (prior_cov_inv @ prior_mean + (1/sigma2) * X.T @ y)

        # 從多元常態分佈中取樣新的 beta
        beta = np.random.multivariate_normal(post_mean, post_cov)

        # b. 根據資料和目前的 beta 更新 sigma2
        # 計算殘差平方和 (Sum of Squared Residuals)
        residuals = y - X @ beta
        ssq_residuals = np.sum(residuals**2)
        
        # 條件後驗分佈的 Inverse-Gamma 參數
        post_alpha = prior_alpha + len(y) / 2
        post_beta = prior_beta + ssq_residuals / 2

        # 從 Inverse-Gamma 分佈中取樣新的 sigma2
        # (這裡利用 Gamma 分佈的倒數來實現)
        sigma2 = 1 / np.random.gamma(shape=post_alpha, scale=1/post_beta)

        # 捨棄 burn-in 樣本
        if i >= n_burnin:
            idx = i - n_burnin
            beta_samples[idx, :] = beta
            sigma2_samples[idx] = sigma2

    return beta_samples, sigma2_samples

# --- 4. 執行並可視化結果 ---
n_samples = 5000
n_burnin = 1000

beta_samples, sigma2_samples = gibbs_sampling_bayesian_regression(X, y, n_samples, n_burnin)

# 顯示參數後驗分佈的直方圖
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(1, 3, figsize=(18, 5))

# beta_0 (截距) 的後驗分佈
ax[0].hist(beta_samples[:, 0], bins=50, density=True, alpha=0.7, color='skyblue')
ax[0].axvline(true_beta_0, color='r', linestyle='--', label=f'真實值: {true_beta_0:.2f}')
ax[0].set_title(r'後驗分佈 of $\beta_0$')
ax[0].set_xlabel(r'$\beta_0$')
ax[0].legend()

# beta_1 (斜率) 的後驗分佈
ax[1].hist(beta_samples[:, 1], bins=50, density=True, alpha=0.7, color='lightgreen')
ax[1].axvline(true_beta_1, color='r', linestyle='--', label=f'真實值: {true_beta_1:.2f}')
ax[1].set_title(r'後驗分佈 of $\beta_1$')
ax[1].set_xlabel(r'$\beta_1$')
ax[1].legend()

# sigma^2 (方差) 的後驗分佈
ax[2].hist(sigma2_samples, bins=50, density=True, alpha=0.7, color='coral')
ax[2].axvline(true_sigma2, color='r', linestyle='--', label=f'真實值: {true_sigma2:.2f}')
ax[2].set_title(r'後驗分佈 of $\sigma^2$')
ax[2].set_xlabel(r'$\sigma^2$')
ax[2].legend()

plt.tight_layout()
plt.show()

# 顯示樣本的平均值作為後驗估計
print("--- 後驗估計 (樣本平均值) ---")
print(f"beta_0 後驗平均值: {np.mean(beta_samples[:, 0]):.2f}, 真實值: {true_beta_0:.2f}")
print(f"beta_1 後驗平均值: {np.mean(beta_samples[:, 1]):.2f}, 真實值: {true_beta_1:.2f}")
print(f"sigma^2 後驗平均值: {np.mean(sigma2_samples):.2f}, 真實值: {true_sigma2:.2f}")