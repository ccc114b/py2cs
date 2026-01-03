import numpy as np
import math
from typing import Union, List, Tuple

# 抽象化 N 維點和向量
# N 維向量或點，維度 D >= 2
VectorND = np.ndarray 
PointND = VectorND

# --- 1. 定義 N 維幾何物件 ---

class LineND:
    """N 維直線：以向量參數式 r(t) = p + t*d 定義"""
    def __init__(self, p: PointND, direction: VectorND):
        if p.shape != direction.shape or p.ndim != 1:
            raise ValueError("Point and direction must be 1D arrays of the same dimension.")
        self.p = p                   # 直線上的已知點
        self.direction = direction   # 方向向量

class Sphere:
    """N 維球體/圓：以球心和半徑定義"""
    def __init__(self, center: PointND, radius: float):
        self.center = center
        self.radius = radius

# --- 2. 核心 N 維向量運算 ---
# 移除 2D 叉積 (cross_product_2d)，因為 N > 3 時無通用定義。

def get_perpendicular_foot_nd(Q: PointND, L: LineND) -> PointND:
    """
    計算 N 維點 Q 到 N 維直線 L 上的垂足 (Pr)。
    這是 N 維空間中最基本的線性幾何操作之一。
    """
    d = L.direction
    p0 = L.p
    
    # 向量 p0Q
    p0Q = Q - p0
    
    # 點積計算參數 t：t = (p0Q . d) / (d . d)
    d_sq = np.dot(d, d)
    
    if np.isclose(d_sq, 0):
        # 方向向量為零
        return p0

    t = np.dot(p0Q, d) / d_sq
    
    # 垂足 Pr = p0 + t * d
    foot = p0 + t * d
    return foot

def get_distance_point_line_nd(Q: PointND, L: LineND) -> float:
    """計算 N 維點 Q 到 N 維直線 L 的最短距離。"""
    Pr = get_perpendicular_foot_nd(Q, L)
    return np.linalg.norm(Q - Pr)

def closest_points_line_line_nd(L1: LineND, L2: LineND) -> Tuple[PointND, PointND]:
    """
    【新函數】計算 N 維空間中，兩條直線之間距離最近的兩個點 (p1_closest, p2_closest)。
    在 N > 2 時，兩直線通常是異面直線，不相交。
    """
    d1, d2 = L1.direction, L2.direction
    p1, p2 = L1.p, L2.p
    
    p12 = p2 - p1
    d1_sq = np.dot(d1, d1)
    d2_sq = np.dot(d2, d2)
    d1_d2 = np.dot(d1, d2)
    
    # 建立一個 2x2 矩陣 A
    A = np.array([
        [d1_sq, -d1_d2],
        [d1_d2, -d2_sq]
    ])
    
    # 建立向量 b
    b = np.array([
        np.dot(p12, d1),
        np.dot(p12, d2)
    ])
    
    # 計算行列式
    det = np.linalg.det(A)
    
    if np.isclose(det, 0):
        # 方向向量平行或反向 (直線平行或重合)
        # 此時只需在 L1 上任取一點，計算到 L2 的垂足即可
        t = np.dot(p2 - p1, d1) / d1_sq if not np.isclose(d1_sq, 0) else 0
        p1_closest = p1 + t * d1
        p2_closest = get_perpendicular_foot_nd(p1_closest, L2) # 隨意取 L1 上一點到 L2 的垂足
        return p1_closest, p2_closest

    # 求解參數 t 和 s
    try:
        ts = np.linalg.solve(A, b)
        t, s = ts[0], ts[1]
        
        p1_closest = p1 + t * d1
        p2_closest = p2 + s * d2
        return p1_closest, p2_closest
    except np.linalg.LinAlgError:
        # 求解失敗，理論上不應該發生，除非 det 極小且不為 0
        return L1.p, L2.p # 返回起點作為備用

def intersect_line_line_nd(L1: LineND, L2: LineND) -> Union[PointND, None]:
    """
    計算 N 維空間中兩直線的**真正交點**。
    如果兩直線不相交，則返回 None。
    """
    p1_closest, p2_closest = closest_points_line_line_nd(L1, L2)
    
    # 檢查最短距離是否接近零
    distance = np.linalg.norm(p1_closest - p2_closest)
    
    if distance < 1e-9:
        # 距離接近零，認為相交，返回中點
        return (p1_closest + p2_closest) / 2
    else:
        return None # 不相交 (異面或平行不重合)

def intersect_line_sphere(L: LineND, S: Sphere) -> List[PointND]:
    """計算直線 L 與 N 維球體 S 的交點 (通用且正確)。"""
    # 核心數學邏輯與 2D 相同：使用垂足法
    Pr = get_perpendicular_foot_nd(S.center, L)
    
    h_sq = np.linalg.norm(Pr - S.center)**2 # 球心到直線的距離平方
    R_sq = S.radius**2
    
    if h_sq > R_sq + 1e-9: 
        return [] # 無交點
    
    L_side = math.sqrt(max(0, R_sq - h_sq)) # 交點到垂足的距離
    
    norm_d = np.linalg.norm(L.direction)
    if np.isclose(norm_d, 0):
        return [] # 方向向量為零
        
    e = L.direction / norm_d # 直線方向單位向量
    
    if np.isclose(L_side, 0):
        return [Pr] # 相切
    else:
        I1 = Pr + L_side * e
        I2 = Pr - L_side * e
        return [I1, I2]

# 註：兩球體相交 (intersect_sphere_sphere) 在 N > 3 時交集為 N-2 維超球面，
# 複雜度大增，單純返回交點不具代表性，因此不放入此通用套件。

# --- 3. N 維變換 (Transformations) ---

class TransformationsND:
    """提供 N 維空間的平移、縮放、旋轉的幾何操作"""

    @staticmethod
    def _get_rotation_matrix(D: int, axis1: int, axis2: int, angle_rad: float) -> np.ndarray:
        """
        【新函數】生成 N 維空間中繞著 (axis1, axis2) 平面旋轉的旋轉矩陣。
        這是 N 維旋轉的通用定義，需要指定旋轉發生的平面。
        """
        if D < 2 or axis1 == axis2 or axis1 >= D or axis2 >= D:
            raise ValueError("Invalid dimensions or axes for rotation.")
            
        R = np.identity(D)
        c = math.cos(angle_rad)
        s = math.sin(angle_rad)
        
        # 旋轉矩陣 R(i, j) 應用於 i-j 平面
        R[axis1, axis1] = c
        R[axis1, axis2] = -s
        R[axis2, axis1] = s
        R[axis2, axis2] = c
        return R

    @staticmethod
    def translate(obj, T: VectorND):
        """對幾何物件進行平移"""
        if isinstance(obj, LineND):
            return LineND(obj.p + T, obj.direction)
        elif isinstance(obj, Sphere):
            return Sphere(obj.center + T, obj.radius)
        elif isinstance(obj, np.ndarray):
            return obj + T
        return obj

    @staticmethod
    def rotate(obj, rotation_matrix: np.ndarray):
        """
        對幾何物件進行 N 維旋轉。
        需要提供 N x N 的旋轉矩陣。
        """
        
        # 定義一個內部輔助函式，處理 PointND 的旋轉
        def rotate_point_nd(P: PointND, R: np.ndarray) -> PointND:
            if P.shape[0] != R.shape[0]:
                raise ValueError("Point dimension must match rotation matrix dimension.")
            return R @ P
            
        if isinstance(obj, LineND):
            return LineND(
                rotate_point_nd(obj.p, rotation_matrix), 
                rotate_point_nd(obj.direction, rotation_matrix)
            )
        elif isinstance(obj, Sphere):
            return Sphere(
                rotate_point_nd(obj.center, rotation_matrix), 
                obj.radius
            )
        elif isinstance(obj, np.ndarray):
            return rotate_point_nd(obj, rotation_matrix)
        return obj

# --- N 維範例執行 (3D) ---
if __name__ == "__main__":
    
    # 3D 空間定義
    dim = 3
    P_A = np.array([1.0, 1.0, 1.0])
    Q_out = np.array([1.0, 7.0, 4.0])
    
    # 直線 L1 (通過 (1, 1, 1) 且方向為 (1, 1, 1))
    Line1 = LineND(p=P_A, direction=np.array([1.0, 1.0, 1.0]))
    
    # 異面直線 L2 (通過 (10, 0, 0) 且方向為 (0, 1, 0))
    Line2 = LineND(p=np.array([10.0, 0.0, 0.0]), direction=np.array([0.0, 1.0, 0.0]))
    
    # 3D 球體
    Sphere1 = Sphere(center=np.array([4.0, 1.0, 1.0]), radius=3.0)

    print(f"--- N={dim} 維幾何運算 ---")

    # 1. 垂足計算
    foot_nd = get_perpendicular_foot_nd(Q_out, Line1) 
    print(f"1. 點 {Q_out} 到直線 L1 的垂足: {foot_nd}") # 應為 [4., 4., 4.]
    
    # 2. 點到線距離
    dist = get_distance_point_line_nd(Q_out, Line1)
    print(f"2. 點 {Q_out} 到直線 L1 的距離: {dist:.4f}") # 應為 norm([0, 3, 3]) = sqrt(18) ≈ 4.2426

    # 3. 兩異面直線最近點
    p1_c, p2_c = closest_points_line_line_nd(Line1, Line2)
    min_dist = np.linalg.norm(p1_c - p2_c)
    print(f"3. 兩異面直線 L1, L2 最近點: L1 上: {p1_c.round(4)}, L2 上: {p2_c.round(4)}") 
    print(f"   最短距離: {min_dist:.4f}") # (應為 9.2376)

    # 4. 兩直線交點 (應為 None，因 L1, L2 異面)
    intersection_LL = intersect_line_line_nd(Line1, Line2)
    print(f"4. 兩直線 L1, L2 交點: {intersection_LL}") # 應為 None

    # 5. 直線與球體交點
    intersection_LS = intersect_line_sphere(Line1, Sphere1)
    print(f"5. 直線 L1 與球體 S1 交點: {intersection_LS} (應為 [])") # 垂足距離 4.24 > 半徑 3，應為 []

    # 6. N 維變換 (3D 繞 X 軸旋轉 90 度)
    # 旋轉 X 軸 (軸 0) 上的 90 度，即在 Y-Z 平面 (軸 1, 2) 旋轉
    R_matrix_x = TransformationsND._get_rotation_matrix(dim, 1, 2, math.radians(90))
    point_rotated = TransformationsND.rotate(P_A, R_matrix_x) # 點 P_A=[1, 1, 1]
    print(f"\n--- 幾何物件變換 (Transformations) ---")
    print(f"6. 點 {P_A} 繞 X 軸旋轉 90 度 (使用 R 矩陣): {point_rotated.round(4)}") # 應為 (1, -1, 1)