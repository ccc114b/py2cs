import numpy as np

def ito_integral_simulation(f_t, w_t, t, n_paths=1, dt=None):
    """
    使用 Euler-Maruyama 方法近似 Ito 積分。

    參數:
    f_t (function): 被積分者，一個接受時間 t 和布朗運動路徑 w_t 的函數。
    w_t (numpy array): 布朗運動的路徑 (W_t)。
    t (numpy array): 時間點。
    n_paths (int): 模擬的布朗運動路徑數量。
    dt (float): 時間步長。如果為 None，則從 t 計算。

    返回:
    numpy array: 近似 Ito 積分的值。
    """
    if dt is None:
        dt = t[1] - t[0]

    # 計算布朗運動的增量 dW_t
    dw = np.diff(w_t, axis=-1, prepend=w_t[..., [0]])

    # 計算被積分者 f(t, W_t)
    f_values = f_t(t, w_t)

    # 執行伊藤積分的離散求和
    # 在每個時間步長 t_i，我們使用 f(t_i, W_{t_i}) 乘以 dW_t = W_{t_{i+1}} - W_{t_i}
    ito_sum = np.sum(f_values[..., :-1] * dw[..., 1:], axis=-1)

    return ito_sum

def test_ito_integral():
    """
    測試 Ito 積分實作，並與解析解進行比較。
    積分: integral from 0 to T of W_t dW_t
    解析解: 0.5 * (W_T^2 - T)
    """
    T = 1.0  # 總時間
    N_steps = 10000  # 模擬的步數
    n_simulations = 1000  # 模擬的次數以得到平均值
    dt = T / N_steps
    t = np.linspace(0, T, N_steps + 1)

    # 生成多條布朗運動路徑
    # 每一列是一條路徑
    dw = np.sqrt(dt) * np.random.randn(n_simulations, N_steps)
    w = np.concatenate([np.zeros((n_simulations, 1)), np.cumsum(dw, axis=1)], axis=1)

    # 定義被積分者 f(t, W_t) = W_t
    f_t = lambda t, w: w

    # 使用模擬函數計算 Ito 積分的近似值
    ito_approx = ito_integral_simulation(f_t, w, t, n_simulations, dt)

    # 計算解析解
    w_T = w[:, -1]
    ito_exact = 0.5 * (w_T**2 - T)

    # 比較平均值
    print(f"Ito 積分模擬的平均值: {np.mean(ito_approx):.4f}")
    print(f"Ito 積分解析解的平均值: {np.mean(ito_exact):.4f}")
    
    # 計算並比較兩者的標準差，確認模擬結果的分散性與解析解接近
    print(f"Ito 積分模擬的標準差: {np.std(ito_approx):.4f}")
    print(f"Ito 積分解析解的標準差: {np.std(ito_exact):.4f}")

    # 比較模擬結果與解析解的平均絕對誤差 (MAE)
    mae = np.mean(np.abs(ito_approx - ito_exact))
    print(f"\n平均絕對誤差 (MAE): {mae:.4f}")

if __name__ == "__main__":
    test_ito_integral()