import numpy as np
import sdeint
import matplotlib.pyplot as plt

# --- 參數設定 ---
S0 = 100.0
K = 100.0    # 履約價 (Strike Price)
r = 0.05
sigma = 0.2
T = 1.0
n_paths = 300  # 模擬路徑數量 (路徑越多越準，但計算越久)
steps = 100

tspan = np.linspace(0, T, steps + 1)

# --- 定義向量化的 Drift 和 Diffusion ---
# 這裡 s 是一個長度為 n_paths 的向量
def f(s, t):
    return r * s

# 注意：sdeint 在處理多維輸入時，g 預設要返回矩陣。
# 為了讓每條路徑有獨立的隨機震盪，我們需要用特殊方式處理，
# 但最簡單的方法是直接利用 sdeint 的 generator 參數或簡單的對角噪聲。
# 在此為了簡單演示，我們定義 g 為對角矩陣形式，這在 sdeint 中計算量較大。
# **實務上：大量路徑模擬通常直接用 NumPy 手寫 Euler-Maruyama 較快，**
# **但既然要用 sdeint，我們可以這樣寫：**

def g(s, t):
    # 這會建立一個對角矩陣，確保路徑之間互不影響
    return np.diag(sigma * s) 

# 初始狀態：1000 個 S0
x0 = np.full(n_paths, S0)

# --- 求解 (這一步會比較慢，因為矩陣運算) ---
# 為了演示 sdeint 功能我們這樣做，實際生產環境建議用 numpy.random
result = sdeint.itoint(f, g, x0, tspan)

# result 的形狀是 (時間步數, 路徑數)
simulated_paths = result

# --- 計算選擇權價格 ---
# 1. 取得最後一步的股價 S_T
ST = simulated_paths[-1, :]

# 2. 計算 Payoff: max(ST - K, 0)
payoffs = np.maximum(ST - K, 0)

# 3. 折現回現在: mean(Payoff) * e^(-rT)
option_price = np.mean(payoffs) * np.exp(-r * T)

print(f"模擬路徑數: {n_paths}")
print(f"BS 模型買權模擬價格: {option_price:.4f}")

# --- 繪圖 (只畫前 10 條以免太亂) ---
plt.figure(figsize=(10, 6))
plt.plot(tspan, simulated_paths[:, :10]) 
plt.title("Black-Scholes Monte Carlo Simulation (First 10 paths)")
plt.xlabel("Time")
plt.ylabel("Price")
plt.show()