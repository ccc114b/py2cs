from scipy.optimize import minimize
import numpy as np

p = np.array([1/2, 1/4, 1/4])
q0 = np.array([1/3, 1/3, 1/3])

# 目標函數
def func(q):
    # 避免 log 錯誤
    q = np.maximum(q, 1e-10)
    return -np.sum(p * np.log2(q))

# 約束條件：sum(q) - 1 = 0
constraints = ({'type': 'eq', 'fun': lambda q: np.sum(q) - 1})

# 變數範圍：0 < q < 1
bounds = [(1e-10, 1) for _ in range(len(p))]

# 執行優化 (SLSQP 方法)
res = minimize(func, q0, method='SLSQP', bounds=bounds, constraints=constraints)

print("Scipy SLSQP Result:")
print("q =", res.x)