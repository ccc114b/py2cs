import numpy as np

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
        if R[j, j] > 1e-10:  # 避免除零
            Q[:, j] = v / R[j, j]
    
    return Q, R

def qr_decomposition(A):
    """QR 分解"""
    return gram_schmidt(A)

def eigenvalue_decomposition(A, max_iter=1000, tol=1e-10):
    """
    使用 QR 算法進行特徵值分解
    
    Parameters:
    A: 方陣 (n x n)
    max_iter: 最大迭代次數
    tol: 收斂容忍度
    
    Returns:
    eigenvalues: 特徵值陣列
    eigenvectors: 特徵向量矩陣 (每一列是一個特徵向量)
    """
    
    if A.shape[0] != A.shape[1]:
        raise ValueError("矩陣必須是方陣")
    
    n = A.shape[0]
    A_k = A.copy().astype(float)
    V = np.eye(n)  # 用來累積特徵向量
    
    # QR 迭代
    for iteration in range(max_iter):
        Q, R = qr_decomposition(A_k)
        A_k_new = R @ Q
        V = V @ Q  # 累積特徵向量變換
        
        # 檢查收斂 - 看下三角部分是否接近零
        off_diagonal_norm = 0
        for i in range(n):
            for j in range(i):
                off_diagonal_norm += abs(A_k_new[i, j])
        
        if off_diagonal_norm < tol:
            break
            
        A_k = A_k_new
    
    # 提取特徵值（對角線元素）
    eigenvalues = np.diag(A_k)
    
    # 特徵向量需要歸一化
    eigenvectors = np.zeros_like(V)
    for i in range(n):
        norm = np.linalg.norm(V[:, i])
        if norm > 1e-10:
            eigenvectors[:, i] = V[:, i] / norm
        else:
            eigenvectors[:, i] = V[:, i]
    
    return eigenvalues, eigenvectors

def verify_decomposition(A, eigenvalues, eigenvectors, tol=1e-8):
    """驗證特徵值分解的正確性"""
    n = A.shape[0]
    print("驗證特徵值分解:")
    
    for i in range(n):
        # 計算 A * v_i
        Av = A @ eigenvectors[:, i]
        # 計算 λ_i * v_i
        lambda_v = eigenvalues[i] * eigenvectors[:, i]
        
        # 計算誤差
        error = np.linalg.norm(Av - lambda_v)
        print(f"特徵值 {i+1}: λ = {eigenvalues[i]:.6f}, 誤差 = {error:.2e}")
        
        if error > tol:
            print(f"  警告: 特徵值 {i+1} 的誤差超過容忍度")

# 測試範例
if __name__ == "__main__":
    # 創建一個測試矩陣
    np.random.seed(42)
    A = np.array([[4, 2, 1],
                  [2, 3, 0],
                  [1, 0, 2]], dtype=float)
    
    print("原始矩陣 A:")
    print(A)
    print()
    
    # 進行特徵值分解
    eigenvals, eigenvecs = eigenvalue_decomposition(A)
    
    print("特徵值:")
    print(eigenvals)
    print()
    
    print("特徵向量 (每一列是一個特徵向量):")
    print(eigenvecs)
    print()
    
    # 驗證結果
    verify_decomposition(A, eigenvals, eigenvecs)
    
    print("\n" + "="*50)
    
    # 與 numpy 的結果比較
    np_eigenvals, np_eigenvecs = np.linalg.eig(A)
    
    print("NumPy 的特徵值:")
    print(np.sort(np_eigenvals)[::-1])  # 降序排列
    
    print("\n我們的特徵值:")
    print(np.sort(eigenvals)[::-1])  # 降序排列
    
    print(f"\n特徵值差異的最大絕對值: {np.max(np.abs(np.sort(eigenvals) - np.sort(np_eigenvals))):.2e}")
