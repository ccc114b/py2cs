import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt

def svd_using_eigendecomposition(A, full_matrices=True):
    """
    使用特徵值分解實現SVD分解
    
    參數:
    A: 輸入矩陣 (m x n)
    full_matrices: 是否返回完整的U和V矩陣
    
    返回:
    U: 左奇異向量矩陣
    S: 奇異值向量
    Vt: 右奇異向量矩陣的轉置
    """
    m, n = A.shape
    
    # 計算A^T A和AA^T
    AtA = A.T @ A  # n x n
    AAt = A @ A.T  # m x m
    
    # 對A^T A進行特徵值分解獲得V和奇異值的平方
    eigenvals_AtA, V = eigh(AtA)
    
    # 排序（scipy的eigh默認升序，我們需要降序）
    idx = np.argsort(eigenvals_AtA)[::-1]
    eigenvals_AtA = eigenvals_AtA[idx]
    V = V[:, idx]
    
    # 計算奇異值（取正平方根，忽略數值誤差導致的負值）
    singular_values = np.sqrt(np.maximum(eigenvals_AtA, 0))
    
    # 計算有效的奇異值個數（非零奇異值）
    rank = np.sum(singular_values > 1e-10)
    
    if full_matrices:
        # 對AA^T進行特徵值分解獲得完整的U
        eigenvals_AAt, U = eigh(AAt)
        idx_U = np.argsort(eigenvals_AAt)[::-1]
        U = U[:, idx_U]
    else:
        # 只計算前rank個左奇異向量
        U = np.zeros((m, rank))
        for i in range(rank):
            if singular_values[i] > 1e-10:
                U[:, i] = (A @ V[:, i]) / singular_values[i]
        
        # 截取相應大小的V
        V = V[:, :rank]
        singular_values = singular_values[:rank]
    
    return U, singular_values, V.T

def compare_with_numpy_svd(A):
    """比較自實現的SVD與numpy的SVD結果"""
    print("原始矩陣 A:")
    print(A)
    print(f"矩陣形狀: {A.shape}")
    print()
    
    # 使用我們的實現
    U1, S1, Vt1 = svd_using_eigendecomposition(A, full_matrices=False)
    
    # 使用numpy的SVD
    U2, S2, Vt2 = np.linalg.svd(A, full_matrices=False)
    
    print("我們的實現:")
    print(f"U shape: {U1.shape}")
    print(f"S shape: {S1.shape}")  
    print(f"Vt shape: {Vt1.shape}")
    print(f"奇異值: {S1}")
    print()
    
    print("NumPy的SVD:")
    print(f"U shape: {U2.shape}")
    print(f"S shape: {S2.shape}")
    print(f"Vt shape: {Vt2.shape}")
    print(f"奇異值: {S2}")
    print()
    
    # 重構矩陣
    A_reconstructed1 = U1 @ np.diag(S1) @ Vt1
    A_reconstructed2 = U2 @ np.diag(S2) @ Vt2
    
    print("重構誤差比較:")
    error1 = np.linalg.norm(A - A_reconstructed1, 'fro')
    error2 = np.linalg.norm(A - A_reconstructed2, 'fro')
    print(f"我們的實現重構誤差: {error1:.2e}")
    print(f"NumPy重構誤差: {error2:.2e}")
    print()
    
    # 奇異值比較
    print("奇異值比較:")
    for i in range(min(len(S1), len(S2))):
        print(f"第{i+1}個奇異值 - 我們的: {S1[i]:.6f}, NumPy: {S2[i]:.6f}, 差異: {abs(S1[i]-S2[i]):.2e}")

def demonstrate_svd_applications():
    """演示SVD的應用"""
    print("="*60)
    print("SVD應用演示")
    print("="*60)
    
    # 1. 低秩近似
    print("1. 低秩近似演示")
    np.random.seed(42)
    A = np.random.randn(6, 4)
    
    U, S, Vt = svd_using_eigendecomposition(A, full_matrices=False)
    
    print(f"原矩陣秩: {np.linalg.matrix_rank(A)}")
    print(f"奇異值: {S}")
    
    # 使用前k個奇異值重構
    for k in [1, 2, 3]:
        A_k = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
        error = np.linalg.norm(A - A_k, 'fro')
        print(f"使用前{k}個奇異值的重構誤差: {error:.4f}")
    
    print()
    
    # 2. 矩陣的偽逆
    print("2. 矩陣偽逆計算")
    B = np.array([[1, 2, 3], 
                  [4, 5, 6]])
    
    U, S, Vt = svd_using_eigendecomposition(B, full_matrices=False)
    
    # 計算偽逆 A+ = V * S^(-1) * U^T
    S_inv = np.zeros_like(S)
    for i in range(len(S)):
        if S[i] > 1e-10:
            S_inv[i] = 1.0 / S[i]
    
    B_pinv = Vt.T @ np.diag(S_inv) @ U.T
    
    print("原矩陣 B:")
    print(B)
    print("計算的偽逆:")
    print(B_pinv)
    print("驗證 B @ B+ @ B:")
    print(B @ B_pinv @ B)
    print("與numpy pinv比較:")
    print(np.linalg.pinv(B))

if __name__ == "__main__":
    # 測試範例
    print("SVD分解測試")
    print("="*60)
    
    # 測試矩陣1: 簡單的3x3矩陣
    print("測試1: 3x3矩陣")
    A1 = np.array([[1, 2, 3],
                   [4, 5, 6], 
                   [7, 8, 9]], dtype=float)
    
    compare_with_numpy_svd(A1)
    print()
    
    # 測試矩陣2: 非方陣
    print("測試2: 4x3矩陣")
    A2 = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9],
                   [10, 11, 12]], dtype=float)
    
    compare_with_numpy_svd(A2)
    print()
    
    # 測試矩陣3: 隨機矩陣
    print("測試3: 5x3隨機矩陣")
    np.random.seed(123)
    A3 = np.random.randn(5, 3)
    
    compare_with_numpy_svd(A3)
    
    # SVD應用演示
    demonstrate_svd_applications()