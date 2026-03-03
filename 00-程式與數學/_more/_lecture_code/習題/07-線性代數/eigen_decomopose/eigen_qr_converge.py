import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Arial Unicode MS'
plt.rcParams['axes.unicode_minus'] = False

def gram_schmidt(A):
    """Gram-Schmidt 正交化過程"""
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    
    for j in range(n):
        v = A[:, j].copy()
        
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A[:, j])
            v -= R[i, j] * Q[:, i]
        
        R[j, j] = np.linalg.norm(v)
        if R[j, j] > 1e-10:
            Q[:, j] = v / R[j, j]
    
    return Q, R

def qr_algorithm_step_by_step(A, max_iter=20):
    """
    逐步展示 QR 算法的收斂過程
    """
    print("=" * 60)
    print("QR 算法收斂過程詳細分析")
    print("=" * 60)
    
    n = A.shape[0]
    A_k = A.copy().astype(float)
    V = np.eye(n)  # 特徵向量累積矩陣
    
    print(f"初始矩陣 A_0:")
    print(f"{A_k}")
    print(f"特徵值 (numpy 參考): {np.sort(np.linalg.eigvals(A))[::-1]}")
    print()
    
    off_diagonal_norms = []
    diagonal_values = []
    
    for k in range(max_iter):
        # QR 分解
        Q, R = gram_schmidt(A_k)
        
        # 計算新的 A
        A_k_new = R @ Q
        
        # 累積特徵向量
        V = V @ Q
        
        # 計算下三角部分的範數 (衡量收斂程度)
        off_diagonal_norm = 0
        for i in range(n):
            for j in range(i):
                off_diagonal_norm += abs(A_k_new[i, j])
        
        off_diagonal_norms.append(off_diagonal_norm)
        diagonal_values.append(np.diag(A_k_new).copy())
        
        print(f"迭代 {k+1}:")
        print(f"  A_{k+1} = ")
        for row in A_k_new:
            print(f"    [{' '.join(f'{x:8.5f}' for x in row)}]")
        print(f"  對角線元素 (近似特徵值): [{' '.join(f'{x:8.5f}' for x in np.diag(A_k_new))}]")
        print(f"  下三角範數: {off_diagonal_norm:.2e}")
        
        # 檢查收斂
        if off_diagonal_norm < 1e-8:
            print(f"\n*** 在第 {k+1} 次迭代收斂! ***")
            break
        
        A_k = A_k_new
        print()
    
    return A_k, V, off_diagonal_norms, diagonal_values

def demonstrate_similarity_preservation(A):
    """
    演示相似變換如何保持特徵值
    """
    print("\n" + "=" * 60)
    print("相似變換與特徵值保持")
    print("=" * 60)
    
    # 第一步 QR 分解
    Q, R = gram_schmidt(A)
    A1 = R @ Q
    
    print("原始矩陣 A:")
    print(A)
    print(f"特徵值: {np.linalg.eigvals(A)}")
    
    print(f"\n第一次 QR 分解:")
    print(f"Q = ")
    print(Q)
    print(f"R = ")
    print(R)
    
    print(f"\nA1 = R @ Q:")
    print(A1)
    print(f"特徵值: {np.linalg.eigvals(A1)}")
    
    # 驗證相似性: A1 = Q^T @ A @ Q
    A1_similarity = Q.T @ A @ Q
    print(f"\n驗證相似變換 Q^T @ A @ Q:")
    print(A1_similarity)
    print(f"與 R@Q 的差異: {np.max(np.abs(A1 - A1_similarity)):.2e}")

def explain_convergence_theory():
    """
    解釋 QR 算法收斂的數學原理
    """
    print("\n" + "=" * 60)
    print("QR 算法收斂原理")
    print("=" * 60)
    
    explanation = """
    QR 算法的收斂原理基於以下數學事實:
    
    1. 相似變換保持特徵值:
       如果 B = P^(-1) * A * P，則 A 和 B 有相同的特徵值
    
    2. QR 分解的性質:
       每次迭代 A_{k+1} = R_k * Q_k = Q_k^T * A_k * Q_k
       這是一個相似變換，所以特徵值保持不變
    
    3. 收斂機制:
       設 A 的特徵值為 λ_1 > λ_2 > ... > λ_n
       
       QR 算法實際上是一種特殊的冪次法 (Power Method) 的推廣:
       - 每次迭代相當於對所有列向量同時應用冪次法
       - 主導特徵值會"浮現"到對角線上
       
    4. 為什麼會形成上三角矩陣:
       - 較大的特徵值會在對角線上方"積累"
       - 較小的特徵值對應的方向會被"壓制"到下三角部分
       - 經過足夠迭代，下三角元素趨於零
    
    5. 收斂速度:
       收斂速度主要取決於特徵值的比率 |λ_{i+1}/λ_i|
       如果特徵值差距很大，收斂會很快
    """
    
    print(explanation)

def visualize_convergence(off_diagonal_norms, diagonal_values):
    """
    視覺化收斂過程
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 左圖: 下三角範數的收斂
    ax1.semilogy(off_diagonal_norms, 'o-', linewidth=2, markersize=6)
    ax1.set_xlabel('迭代次數')
    ax1.set_ylabel('下三角元素範數 (log scale)')
    ax1.set_title('收斂速度: 下三角元素趨於零')
    ax1.grid(True, alpha=0.3)
    
    # 右圖: 對角線元素(特徵值)的收斂
    diagonal_array = np.array(diagonal_values)
    for i in range(diagonal_array.shape[1]):
        ax2.plot(diagonal_array[:, i], 'o-', label=f'λ_{i+1}', linewidth=2, markersize=4)
    
    ax2.set_xlabel('迭代次數')
    ax2.set_ylabel('對角線元素值')
    ax2.set_title('特徵值收斂過程')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# 主演示程序
if __name__ == "__main__":
    # 使用一個具有明顯不同特徵值的矩陣
    A = np.array([[6, 2, 1],
                  [2, 3, 1],
                  [1, 1, 1]], dtype=float)
    
    print("使用矩陣:")
    print(A)
    print(f"真實特徵值 (NumPy): {np.sort(np.linalg.eigvals(A))[::-1]}")
    
    # 逐步演示 QR 算法
    final_A, V, norms, diag_values = qr_algorithm_step_by_step(A, max_iter=15)
    
    # 演示相似變換
    demonstrate_similarity_preservation(A)
    
    # 解釋理論
    explain_convergence_theory()
    
    # 視覺化 (如果在支援 matplotlib 的環境中)
    try:
        visualize_convergence(norms, diag_values)
    except:
        print("\n(圖形化顯示需要 matplotlib 支援)")
    
    print("\n" + "=" * 60)
    print("總結:")
    print("QR 算法通過反覆的相似變換，將矩陣逐漸轉化為上三角形式，")
    print("其對角線元素收斂到特徵值，而累積的變換矩陣給出特徵向量。")
    print("=" * 60)