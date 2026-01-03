import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.mixture import GaussianMixture

# ----------------------------------------------------------------
# 1. 最大似然估計 (Maximum Likelihood Estimation, MLE)
# ----------------------------------------------------------------

def estimate_mle_bernoulli(data):
    """
    估計伯努利分佈的參數 theta (正面機率)
    """
    n = len(data)
    k = np.sum(data)
    theta_hat = k / n
    return theta_hat

# ----------------------------------------------------------------
# 2. 潛在變量模型：高斯混合模型 (Gaussian Mixture Model)
# ----------------------------------------------------------------

def train_gmm(data, n_components=2):
    """
    使用 EM 演算法訓練 GMM，這是一個典型的潛在變量模型
    """
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(data)
    return gmm

# ----------------------------------------------------------------
# 3. 資訊理論：KL 散度 (KL Divergence)
# ----------------------------------------------------------------

def calculate_kl_divergence(p_probs, q_probs):
    """
    計算兩個離散機率分佈 P 與 Q 之間的 KL 散度
    """
    # 確保分佈總和為 1
    p = p_probs / np.sum(p_probs)
    q = q_probs / np.sum(q_probs)
    
    # 避免 log(0)
    kl_div = np.sum(p * np.log(p / q + 1e-10))
    return kl_div

# --- 主程式執行 ---

if __name__ == "__main__":
    # 範例 1: MLE 估計
    # 模擬投擲硬幣：1 代表正面，0 代表反面
    coin_flips = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 1])
    theta = estimate_mle_bernoulli(coin_flips)
    print(f"MLE 估計的正面機率 theta: {theta:.2f}")

    # 範例 2: GMM 生成模型
    # 產生兩組不同均值的高斯數據
    data1 = np.random.normal(loc=-2, scale=0.5, size=(200, 1))
    data2 = np.random.normal(loc=3, scale=1.2, size=(300, 1))
    X = np.vstack([data1, data2])

    model = train_gmm(X, n_components=2)
    
    # 從模型中抽樣產生新數據 (Sampling)
    X_new, y_new = model.sample(500)
    
    # 範例 3: KL 散度
    # 比較兩個正態分佈的差異
    x_axis = np.linspace(-5, 5, 100)
    p_dist = norm.pdf(x_axis, 0, 1)    # P ~ N(0, 1)
    q_dist = norm.pdf(x_axis, 0.5, 1.5) # Q ~ N(0.5, 1.5)
    
    kl = calculate_kl_divergence(p_dist, q_dist)
    print(f"P 與 Q 之間的 KL 散度: {kl:.4f}")

    # 繪製 GMM 生成結果
    plt.figure(figsize=(10, 4))
    plt.hist(X, bins=50, density=True, alpha=0.5, label='Original Data')
    plt.hist(X_new, bins=50, density=True, alpha=0.5, color='red', label='Generated Data')
    plt.title("GMM: Original vs Generated Data Distribution")
    plt.legend()
    plt.show()