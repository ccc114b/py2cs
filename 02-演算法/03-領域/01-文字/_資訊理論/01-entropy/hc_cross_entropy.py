import math
import numpy as np

np.set_printoptions(precision=4, suppress=True)

# ---------------------------------------------------------
# 1. 定義 Cross Entropy
# ---------------------------------------------------------
def log2(x):
    # 防止 log(0) 錯誤
    return math.log(max(x, 1e-15), 2)

def cross_entropy(p, q):
    r = 0
    for i in range(len(p)):
        r += p[i] * log2(1 / q[i])
    return r

def entropy(p):
    return cross_entropy(p, p)

# ---------------------------------------------------------
# 2. 隨機爬山演算法 (Random Hill Climbing)
# ---------------------------------------------------------
def random_hill_climbing(loss_func, p_target, q_init, max_loops=10000):
    
    # 複製初始狀態
    q = q_init.copy()
    current_loss = loss_func(p_target, q)
    
    # 初始搜尋半徑 (步長)
    step_size = 0.5
    
    print(f"Start Random Search...")
    print(f"Initial q: {q}, Loss: {current_loss:.5f}")
    
    for i in range(max_loops):
        # 1. 產生隨機擾動 (Random Noise)
        # 產生一個跟 q 一樣長的隨機向量，數值在 -1 到 1 之間
        noise = np.random.uniform(-1, 1, size=q.shape)
        
        # 2. 試探新的點
        q_try = q + (noise * step_size)
        
        # --- 解決障礙：約束處理 ---
        # (A) 確保所有數值 > 0 (取絕對值或 max)
        q_try = np.abs(q_try) + 1e-10 
        
        # (B) 歸一化：確保總和為 1
        q_try = q_try / np.sum(q_try)
        
        # 3. 計算新 Loss
        new_loss = loss_func(p_target, q_try)
        
        # 4. 決策：如果新 Loss 比較小，就移動過去 (貪婪策略)
        if new_loss < current_loss:
            q = q_try
            current_loss = new_loss
        
        # 5. 動態調整步長 (Learning Rate Decay)
        # 隨著時間過去，縮小搜尋範圍，讓結果更精確
        if i % 1000 == 0:
            step_size *= 0.8 # 每次縮小 20%
            print(f"{i:05d}: Loss={current_loss:.5f} q={q} step={step_size:.5f}")
            
    return q

# ---------------------------------------------------------
# 3. 主程式驗證
# ---------------------------------------------------------
if __name__ == "__main__":
    # 目標分佈 p
    #p = np.array([1/2, 1/4, 1/4])
    p = np.array([0.2, 0.5, 0.3])
    
    # 初始猜測 q
    #q_start = np.array([1/3, 1/3, 1/3])
    q_start = np.array([0.1, 0.7, 0.2])
    
    print(f"Target p: {p}")
    print(f"Target Min Loss (Entropy): {entropy(p):.5f}\n")
    
    # 執行優化
    q_final = random_hill_climbing(cross_entropy, p, q_start, max_loops=20000)
    
    print("-" * 60)
    print("Final Result:")
    print(f"Optimized q : {q_final}")
    print(f"Target    p : {p}")
    print(f"Final Loss  : {cross_entropy(p, q_final):.5f}")
    
    # 驗證
    error = np.sum(np.abs(q_final - p))
    if error < 0.01:
        print("\n✅ 驗證成功：在不使用梯度的情況下，隨機搜尋法找到了 q=p")
    else:
        print("\n⚠️ 未完全收斂")