import numpy as np
import matplotlib.pyplot as plt

# 設定隨機種子，確保每次執行結果一致
np.random.seed(42)

def verify_central_limit_theorem(num_samples=10000, sample_size_small=5, sample_size_large=30):
    """
    透過繪製統計圖來驗證中央極限定理。

    Args:
        num_samples (int): 模擬抽樣的次數。
        sample_size_small (int): 第一次模擬時，每次抽樣的樣本數（較小）。
        sample_size_large (int): 第二次模擬時，每次抽樣的樣本數（較大）。
    """
    # 1. 選擇一個非常態分布的母體：均勻分布
    # 我們假設母體數據介於 0 到 1 之間
    population = np.random.uniform(0, 1, 100000)

    # 2.3. 模擬多次抽樣並計算平均數
    sample_means_small = []
    sample_means_large = []
    
    for _ in range(num_samples):
        # 每次從母體中隨機抽取 'sample_size_small' 個樣本，並計算其平均數
        sample_small = np.random.choice(population, size=sample_size_small)
        sample_means_small.append(np.mean(sample_small))
        
        # 每次從母體中隨機抽取 'sample_size_large' 個樣本，並計算其平均數
        sample_large = np.random.choice(population, size=sample_size_large)
        sample_means_large.append(np.mean(sample_large))

    # 4.5. 繪製統計圖來展示
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('中央極限定理 (Central Limit Theorem) 驗證', fontsize=16)

    # 第一張圖：母體分布
    axes[0].hist(population, bins=50, density=True, color='skyblue', edgecolor='black')
    axes[0].set_title('母體 (Population) 分布\n(均勻分布)', fontsize=14)
    axes[0].set_xlabel('數值')
    axes[0].set_ylabel('頻率')
    
    # 第二張圖：小樣本數的平均數分布
    axes[1].hist(sample_means_small, bins=50, density=True, color='lightgreen', edgecolor='black')
    axes[1].set_title(f'樣本平均數分布 (每次抽樣 N={sample_size_small})', fontsize=14)
    axes[1].set_xlabel('樣本平均數')
    axes[1].set_ylabel('頻率')
    
    # 第三張圖：大樣本數的平均數分布
    axes[2].hist(sample_means_large, bins=50, density=True, color='salmon', edgecolor='black')
    axes[2].set_title(f'樣本平均數分布 (每次抽樣 N={sample_size_large})', fontsize=14)
    axes[2].set_xlabel('樣本平均數')
    axes[2].set_ylabel('頻率')
    
    # 在第三張圖上疊加一個常態分布曲線，以作對比
    mu = np.mean(sample_means_large)
    sigma = np.std(sample_means_large)
    x = np.linspace(min(sample_means_large), max(sample_means_large), 100)
    pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma)**2)
    axes[2].plot(x, pdf, color='red', linestyle='--', linewidth=2, label='常態分布曲線')
    axes[2].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # 調整子圖佈局
    plt.show()

if __name__ == "__main__":
    verify_central_limit_theorem()