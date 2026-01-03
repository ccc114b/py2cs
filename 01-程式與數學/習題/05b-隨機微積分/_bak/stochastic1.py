import numpy as np
import matplotlib.pyplot as plt
from stochastic.processes.continuous import GeometricBrownianMotion

# --- 參數設定 ---
mu = 0.05      # 漂移率 (Drift), e.g., 年化報酬率 5%
sigma = 0.2    # 波動率 (Scale/Volatility), e.g., 年化波動率 20%
s0 = 100       # 初始價格 (Initial Value)
t = 1.0        # 總時間長度 (Time period in years)
n_steps = 252  # 模擬的步數 (Steps), e.g., 一年的交易日

# 1. 實例化 GeometricBrownianMotion
# 初始值 S0 作為 'x0' 參數傳入
gbm = GeometricBrownianMotion(drift=mu, scale=sigma, t=t, x0=s0)

# 2. 生成一條隨機路徑 (Sample a path)
# n_steps 是指過程中的時間點數量 (包括 t=0)
path = gbm.sample(n_steps=n_steps)

# 3. 獲取時間軸，用於繪圖
# sample 方法會自動生成等間隔的時間點
time = np.linspace(0, t, n_steps)

# 4. 繪製結果
plt.figure(figsize=(10, 6))
plt.plot(time, path)
plt.title(
    f'幾何布朗運動模擬 (GBM)\n'
    f'μ={mu}, σ={sigma}, S₀={s0}'
)
plt.xlabel('時間 (年)')
plt.ylabel('資產價格')
plt.grid(True, alpha=0.5)
plt.show()