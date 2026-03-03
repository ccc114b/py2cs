import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# --- 1. 設定目標分佈 (雙峰常態混合分佈) ---
# 這個函數代表我們的目標分佈的相對機率密度 (未歸一化)
def target_pdf(x):
    """
    目標分佈的機率密度函數，由兩個常態分佈混合而成。
    """
    # 混合分佈的第一個分量
    comp1 = 0.3 * norm.pdf(x, loc=-3, scale=1)
    # 混合分佈的第二個分量
    comp2 = 0.7 * norm.pdf(x, loc=5, scale=np.sqrt(2))
    return comp1 + comp2

# --- 2. 設定提議分佈 (常態分佈) ---
# 提議分佈的標準差 (超參數)
proposal_std = 3.0

def proposal_dist(current_x):
    """
    根據當前位置，從常態分佈中生成一個候選樣本。
    """
    return np.random.normal(current_x, proposal_std)

# 因為提議分佈是對稱的，所以 Q(from|to) / Q(to|from) = 1。
# 因此，這裡的 proposal_pdf 函數實際上可以省略，但為了完整性而保留。
def proposal_pdf(from_x, to_x):
    return norm.pdf(to_x, loc=from_x, scale=proposal_std)

# --- 3. 執行 Metropolis-Hastings 演算法 ---
def metropolis_hastings(n_samples, n_burnin):
    """
    使用 M-H 演算法從目標分佈中取樣。
    """
    # 隨機初始化一個起始點
    current_x = np.random.uniform(-10, 10)
    
    samples = []
    
    for i in range(n_samples + n_burnin):
        # 產生一個候選樣本
        proposed_x = proposal_dist(current_x)

        # 計算接受率
        # 因為提議分佈是對稱的，所以 proposal_ratio = 1，接受率公式簡化為：
        # acceptance_ratio = min(1, target_pdf(proposed_x) / target_pdf(current_x))
        acceptance_ratio = min(1, target_pdf(proposed_x) / target_pdf(current_x))

        # 決定是否接受新樣本
        if np.random.uniform(0, 1) < acceptance_ratio:
            current_x = proposed_x
        
        # 儲存樣本 (捨棄 burn-in 階段)
        if i >= n_burnin:
            samples.append(current_x)
            
    return np.array(samples)

# --- 4. 執行並可視化結果 ---
n_samples_val = 50000
n_burnin_val = 10000
samples = metropolis_hastings(n_samples_val, n_burnin_val)

# 繪製結果
plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(10, 6))

# 繪製 M-H 樣本的直方圖
plt.hist(samples, bins=100, density=True, alpha=0.6, label='Metropolis-Hastings 樣本分佈', color='coral')

# 繪製真實的目標分佈 PDF 作為比較
x_values = np.linspace(-10, 10, 1000)
plt.plot(x_values, target_pdf(x_values), 'b-', lw=2, label='真實目標 PDF')

plt.title(f'Metropolis-Hastings 從雙峰分佈中取樣 (proposal_std={proposal_std})', fontsize=16)
plt.xlabel('x', fontsize=12)
plt.ylabel('機率密度', fontsize=12)
plt.legend(fontsize=12)
plt.show()

# 檢查樣本統計量
print(f"樣本平均值: {np.mean(samples):.2f}")
print(f"樣本標準差: {np.std(samples):.2f}")