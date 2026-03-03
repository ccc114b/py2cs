import numpy as np
import sdeint
import matplotlib.pyplot as plt

# 定義參數
mu = 0.1
sigma = 0.2
x0 = 1.0  # 初始值
tspan = np.linspace(0, 10, 1001) # 時間區間

# 定義漂移項 (Drift, f) 和 擴散項 (Diffusion, g)
# 方程形式: dX = f(x,t)dt + g(x,t)dW
def f(x, t):
    return mu * x

def g(x, t):
    return sigma * x

# 求解
result = sdeint.itoint(f, g, x0, tspan)

# 繪圖
plt.plot(tspan, result)
plt.title("Geometric Brownian Motion Simulation")
plt.xlabel("Time")
plt.ylabel("X(t)")
plt.show()