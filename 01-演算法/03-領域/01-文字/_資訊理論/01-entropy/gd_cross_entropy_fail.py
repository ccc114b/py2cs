import math
import numpy as np
from numpy.linalg import norm

# 設定 numpy 浮點數顯示精度
np.set_printoptions(precision=4, suppress=True)

# ---------------------------------------------------------
# 1. 定義數學與 Loss 函數
# ---------------------------------------------------------

def log2(x):
    # 加上極小值 1e-15 防止 log(0) 錯誤
    return math.log(max(x, 1e-15), 2)

def cross_entropy(p, q):
    r = 0
    for i in range(len(p)):
        r += p[i] * log2(1 / q[i])
    return r

def entropy(p):
    return cross_entropy(p, p)

# ---------------------------------------------------------
# 2. 數值梯度工具 (基於你提供的程式碼)
# ---------------------------------------------------------

# 函數 f 對變數 k 的偏微分: df / dk
def df(f, p, k, step=0.0001):
    p1 = p.copy()
    p1[k] = p[k] + step
    # 注意：這裡計算微分時，p1 暫時不需要歸一化，
    # 因為我們只是要看「如果這個分量增加一點點，函數值怎麼變」
    return (f(p1) - f(p)) / step

# 函數 f 在點 p 上的梯度
def grad(f, p, step=0.0001):
    gp = p.copy()
    for k in range(len(p)):
        gp[k] = df(f, p, k, step)
    return gp

# ---------------------------------------------------------
# 3. 梯度下降法 (修改版：加入歸一化機制)
# ---------------------------------------------------------

def gradientDescendent(f, p0, step=0.01, max_loops=5000, dump_period=500):
    p = p0.copy()
    fp0 = f(p)
    
    print(f"Start Optimization...")
    print(f"Initial p: {p}")
    print(f"Initial Loss: {fp0:.5f}")
    print("-" * 60)

    for i in range(max_loops):
        fp = f(p)
        
        # 1. 計算數值梯度
        gp = grad(f, p) 
        glen = norm(gp) 
        
        if i % dump_period == 0: 
            print('{:05d}: Loss={:.5f} q={:s} |grad|={:.5f}'.format(i, fp, str(p), glen))
        
        # 停止條件：梯度極小 (注意：因為我們後面有強制歸一化，梯度可能不會完全變成0，但會很小)
        if glen < 0.00001: 
            break
            
        # 2. 往梯度的反方向走 (更新參數)
        gstep = np.multiply(gp, -1 * step) 
        p += gstep 
        
        # ---------------------------------------------------
        # 3. 解決障礙：Projected Gradient Descent (投影與歸一化)
        # ---------------------------------------------------
        
        # (A) 確保數值大於 0 (避免 log 爆掉)
        p = np.maximum(p, 1e-10)
        
        # (B) 歸一化：強制總和為 1
        p = p / np.sum(p)
        
        fp0 = fp

    print("-" * 60)
    print('{:05d}: Loss={:.5f} q={:s} |grad|={:.5f}'.format(i, fp, str(p), glen))
    return p 

# ---------------------------------------------------------
# 4. 主程式驗證
# ---------------------------------------------------------

if __name__ == "__main__":
    # 目標分佈 P (固定)
    P_target = np.array([1/2, 1/4, 1/4])
    
    # 初始分佈 Q (可調變數，起始值設為均勻分佈)
    Q_init   = np.array([1/3, 1/3, 1/3])

    # 為了配合 gradientDescendent 的介面，我們定義一個 lambda 或 函數 wrapper
    # 這個函數只接受 q 作為輸入 (因為 p 是固定的)
    def loss_function(q):
        return cross_entropy(P_target, q)

    # 理論最小值
    min_val = entropy(P_target)
    print(f"Target P (Truth) : {P_target}")
    print(f"Theoretical Min Loss (Entropy): {min_val:.5f}\n")

    # 執行優化
    # step 建議設小一點，因為機率數值很敏感 (0~1之間)
    Q_final = gradientDescendent(loss_function, Q_init, step=0.05, max_loops=5000, dump_period=1000)

    print("\n[驗證結果]")
    print(f"Final Optimized Q: {Q_final}")
    print(f"Target P         : {P_target}")
    
    # 計算誤差
    error = np.sum(np.abs(Q_final - P_target))
    if error < 0.01:
        print("✅ 成功：Q 已收斂至 P")
    else:
        print("❌ 失敗：Q 尚未完全收斂")