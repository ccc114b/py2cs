import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# --- 1. 設定目標分佈 (Beta 分佈) ---
# 目標分佈參數
alpha_target = 2
beta_target = 5
target_dist = beta(alpha_target, beta_target)

# 我們只需要知道目標分佈的相對機率，這裡就是 Beta 分佈的 PDF
def target_pdf(x):
    if 0 <= x <= 1:
        return target_dist.pdf(x)
    return 0

# --- 2. 設定提議分佈 (常態分佈) ---
# 標準差 (一個重要的超參數)
proposal_std = 0.1

def proposal_dist(current_x):
    # 以當前 x 為平均值，從常態分佈中取樣
    return np.random.normal(current_x, proposal_std)

def proposal_pdf(from_x, to_x):
    # 常態分佈的 PDF，用於計算接受率
    return (1 / (proposal_std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((to_x - from_x) / proposal_std)**2)

# --- 3. 執行 Metropolis-Hastings 演算法 ---
def metropolis_hastings(n_samples, n_burnin):
    # 初始化一個隨機的起始點，範圍在 [0, 1]
    current_x = np.random.uniform(0, 1)
    
    samples = []
    
    for i in range(n_samples + n_burnin):
        # 產生一個候選樣本
        proposed_x = proposal_dist(current_x)

        # 計算接受率
        # M-H 的核心：計算目標分佈和提議分佈的相對比率
        # P(x_prime) / P(x_current)
        prob_ratio = target_pdf(proposed_x) / target_pdf(current_x)
        
        # Q(x_current | x_prime) / Q(x_prime | x_current)
        proposal_ratio = proposal_pdf(proposed_x, current_x) / proposal_pdf(current_x, proposed_x)
        
        acceptance_ratio = min(1, prob_ratio * proposal_ratio)

        # 決定是否接受
        if np.random.uniform(0, 1) < acceptance_ratio:
            current_x = proposed_x
        
        # 儲存樣本 (捨棄 burn-in 階段)
        if i >= n_burnin:
            samples.append(current_x)
            
    return np.array(samples)

# --- 4. 執行並可視化結果 ---
n_samples_val = 20000
n_burnin_val = 5000
samples = metropolis_hastings(n_samples_val, n_burnin_val)

# 繪製結果
plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(10, 6))

# 繪製 M-H 樣本的直方圖
plt.hist(samples, bins=50, density=True, alpha=0.6, label='Metropolis-Hastings 樣本分佈', color='skyblue')

# 繪製真實的 Beta 分佈 PDF 作為比較
x_values = np.linspace(0, 1, 1000)
plt.plot(x_values, target_dist.pdf(x_values), 'r-', lw=2, label='真實 Beta(2, 5) PDF')

plt.title('Metropolis-Hastings 從 Beta 分佈中取樣', fontsize=16)
plt.xlabel('x', fontsize=12)
plt.ylabel('機率密度', fontsize=12)
plt.legend(fontsize=12)
plt.show()

# 檢查樣本統計量
print(f"樣本平均值: {np.mean(samples):.4f}")
print(f"理論平均值: {alpha_target / (alpha_target + beta_target):.4f}")