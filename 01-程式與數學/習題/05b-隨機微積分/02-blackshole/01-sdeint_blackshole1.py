import numpy as np
import sdeint
import matplotlib.pyplot as plt

# --- 1. 設定 Black-Scholes 參數 ---
S0 = 100.0   # 初始股價
r = 0.05     # 無風險利率 (Drift term for risk-neutral pricing)
sigma = 0.2  # 波動率 (20%)
T = 1.0      # 到期時間 (1年)
steps = 252  # 模擬步數 (假設一年 252 個交易日)

# 設定時間區間
tspan = np.linspace(0, T, steps + 1)

# --- 2. 定義 SDE 的係數 ---
# 方程: dS = f(S, t)dt + g(S, t)dW

# 漂移項 (Drift): r * S
def f(s, t):
    return r * s

# 擴散項 (Diffusion): sigma * S
def g(s, t):
    return sigma * s

# --- 3. 求解 SDE ---
# sdeint.itoint 適用於 Ito 積分 (Black-Scholes 預設是 Ito Calculus)
result = sdeint.itoint(f, g, S0, tspan)

# --- 4. 繪圖 ---
plt.figure(figsize=(10, 6))
plt.plot(tspan, result, label='Simulated Path (GBM)')
plt.axhline(S0, color='r', linestyle='--', alpha=0.5, label='Start Price')
plt.title(f"Black-Scholes Simulation\n$S_0={S0}, r={r}, \sigma={sigma}$")
plt.xlabel("Time (Year)")
plt.ylabel("Stock Price ($S_t$)")
plt.legend()
plt.grid(True)
plt.show()