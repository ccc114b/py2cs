import numpy as np
import matplotlib.pyplot as plt

# 設定隨機種子，確保每次執行結果一致
np.random.seed(42)

# 確保你的 Mac 系統有 PingFang.ttc 這個字型，
# 如果沒有，請換成其他你確認可用的中文字型名稱，如 'Arial Unicode MS'
plt.rcParams['font.family'] = 'Arial Unicode MS'

# 解決負號無法正常顯示的問題
plt.rcParams['axes.unicode_minus'] = False

def get_population(distribution_type, size):
    """
    根據指定的分布類型生成母體數據。

    Args:
        distribution_type (str): 母體分布類型，可選 'uniform', 'bernoulli', 'normal', 'exponential'。
        size (int): 母體數據的大小。

    Returns:
        numpy.ndarray: 生成的母體數據。
    """
    if distribution_type == 'uniform':
        # 均勻分布 (類似擲骰子，但連續)
        # 數據介於 0 到 1 之間，機率密度是平坦的
        return np.random.uniform(0, 1, size)
    
    elif distribution_type == 'bernoulli':
        # 伯努利分布 (類似丟銅板)
        # 數據只有 0 或 1，這裡設定 0.5 的機率出現 1
        return np.random.choice([0, 1], size=size, p=[0.5, 0.5])
    
    elif distribution_type == 'normal':
        # 常態分布 (鐘形曲線)
        # 平均值為 0，標準差為 1
        return np.random.normal(loc=0, scale=1, size=size)
        
    elif distribution_type == 'exponential':
        # 指數分布 (不對稱的分布)
        # 參數 scale=1，代表平均值為 1
        return np.random.exponential(scale=1, size=size)
    
    else:
        raise ValueError("Invalid distribution_type. Choose from 'uniform', 'bernoulli', 'normal', 'exponential'.")

def verify_central_limit_theorem(distribution_type, num_samples=10000, sample_sizes=[1, 2, 10, 30]):
    """
    透過繪製統計圖來驗證中央極限定理，並可選擇不同的母體分布。

    Args:
        distribution_type (str): 母體分布類型。
        num_samples (int): 模擬抽樣的次數。
        sample_sizes (list): 每次抽樣的樣本數列表。
    """
    # 1. 根據指定類型生成母體
    population = get_population(distribution_type, 100000)

    # 4. 繪製統計圖來展示
    # 調整 figsize 的大小，使圖表更緊湊
    fig, axes = plt.subplots(2, 2, figsize=(5, 5))
    fig.suptitle(f'中央極限定理驗證 (母體為 {distribution_type} 分布)', fontsize=12)
    
    # 將 axes 陣列扁平化，方便迭代
    axes = axes.flatten()

    for i, sample_size in enumerate(sample_sizes):
        # 2.3. 模擬多次抽樣並計算平均數
        sample_means = []
        for _ in range(num_samples):
            sample = np.random.choice(population, size=sample_size)
            sample_means.append(np.mean(sample))

        # 5. 繪製樣本平均數分布圖
        ax = axes[i]
        ax.hist(sample_means, bins=50, density=True, color='skyblue', edgecolor='black')
        ax.set_title(f'N={sample_size}', fontsize=12) # 簡化標題
        ax.set_xlabel('樣本平均數', fontsize=10)
        ax.set_ylabel('頻率', fontsize=10)
        
        # 在圖上疊加一個常態分布曲線，以作對比
        mu = np.mean(sample_means)
        sigma = np.std(sample_means)
        x = np.linspace(min(sample_means), max(sample_means), 100)
        pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma)**2)
        ax.plot(x, pdf, color='red', linestyle='--', linewidth=2, label='常態分布曲線')
        ax.legend(fontsize=8) # 調整圖例字體大小
        
    # 調整子圖間距，讓它們更緊密
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    # 執行不同分布的驗證
    print("--- 驗證均勻分布 ---")
    verify_central_limit_theorem('uniform')
    
    print("--- 驗證伯努利分布 (丟銅板) ---")
    verify_central_limit_theorem('bernoulli')
    
    print("--- 驗證常態分布 ---")
    verify_central_limit_theorem('normal')
    
    print("--- 驗證指數分布 ---")
    verify_central_limit_theorem('exponential')