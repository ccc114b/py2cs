import numpy as np
import matplotlib.pyplot as plt

def geometric_brownian_motion(S0, mu, sigma, T, n_steps):
    """
    使用歐拉-丸山法模擬幾何布朗運動
    
    參數:
    S0 (float): 初始資產價格
    mu (float): 漂移率 (年化)
    sigma (float): 波動率 (年化)
    T (float): 模擬總時長 (年)
    n_steps (int): 模擬的步數
    
    回傳:
    np.ndarray: 包含資產價格隨時間變化的陣列
    """
    dt = T / n_steps
    t = np.linspace(0, T, n_steps + 1)
    
    # 隨機數生成
    dW = np.random.normal(loc=0.0, scale=np.sqrt(dt), size=n_steps)
    
    # 初始化資產價格陣列
    S = np.zeros(n_steps + 1)
    S[0] = S0
    
    # 執行歐拉-丸山法模擬
    for i in range(n_steps):
        dS = mu * S[i] * dt + sigma * S[i] * dW[i]
        S[i+1] = S[i] + dS
    
    return t, S

# --- 測試與繪圖 ---
if __name__ == "__main__":
    # 設定參數
    S0 = 100.0        # 初始價格
    mu = 0.05         # 漂移率 (5%)
    sigma = 0.20      # 波動率 (20%)
    T = 1.0           # 模擬一年
    n_steps = 252 * 5 # 模擬5年的交易日 (每年252天)
    
    # 執行模擬並繪圖
    num_paths = 5     # 模擬5條不同的路徑
    plt.figure(figsize=(10, 6))
    
    for i in range(num_paths):
        t, S_path = geometric_brownian_motion(S0, mu, sigma, T, n_steps)
        plt.plot(t, S_path, label=f'Path {i+1}')

    plt.title('Simulated Geometric Brownian Motion Paths (Ito Integration)')
    plt.xlabel('Time (Years)')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.grid(True)
    plt.show()