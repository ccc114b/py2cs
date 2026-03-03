import numpy as np
import sdeint
import matplotlib.pyplot as plt

# 定義一個簡單的 Ito SDE: dX = -X dt + dW (Ornstein-Uhlenbeck Process)
# 漂移項 (Drift)
def f(x, t):
    return -1.0 * x

# 擴散項 (Diffusion)
# 在 Ito 積分中，這代表 dW 前面的係數
def g(x, t):
    return 1.0

x0 = 10.0
tspan = np.linspace(0, 5, 501)

# 明確呼叫 'itoint'，代表我們認定這是 Ito 形式的方程
result = sdeint.itoint(f, g, x0, tspan)

plt.plot(tspan, result)
plt.title("Solution using sdeint.itoint (Ito Integral)")
plt.xlabel("Time")
plt.ylabel("X(t)")
plt.show()