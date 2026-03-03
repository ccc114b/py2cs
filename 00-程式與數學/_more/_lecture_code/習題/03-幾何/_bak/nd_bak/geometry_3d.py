import numpy as np
import math
from typing import Union, List, Tuple

from geometry_nd import (
    VectorND, PointND, LineND, Sphere, 
    get_perpendicular_foot_nd, intersect_line_line_nd, intersect_line_sphere,
    TransformationsND
)

# --- 1. 3D 幾何物件定義 (繼承/別名) ---

class Line3D(LineND):
    """3D 直線：基於 LineND，強制維度 D=3"""
    def __init__(self, p: PointND, direction: VectorND):
        if p.shape[0] != 3 or direction.shape[0] != 3:
            raise ValueError("3D Line must be initialized with 3D vectors.")
        super().__init__(p, direction)

class Sphere3D(Sphere):
    """3D 球體：基於 Sphere，強制維度 D=3"""
    def __init__(self, center: PointND, radius: float):
        if center.shape[0] != 3:
            raise ValueError("3D Sphere center must be a 3D vector.")
        super().__init__(center, radius)

# 為了方便使用，建立別名
Line = Line3D
Sphere = Sphere3D

# --- 2. 3D 專屬向量運算 ---

def cross_product_3d(v1: VectorND, v2: VectorND) -> VectorND:
    """【3D 專屬】計算 3D 向量的叉積 (返回 3D 向量)"""
    if v1.shape[0] != 3 or v2.shape[0] != 3:
        raise ValueError("Cross product is only defined for 3D vectors in this context.")
    # 使用 numpy 內建的 3D 叉積
    return np.cross(v1, v2)

# --- 3. 3D 專屬相交運算 (使用 ND 基礎) ---

def intersect_line_line(L1: Line, L2: Line) -> Union[PointND, None]:
    """
    【簡化/別名】計算兩直線交點。
    直接使用 ND 版本，這是查找異面直線是否存在交點的通用方法。
    """
    # 假設 ND 模組中已包含 closest_points_line_line_nd 和 intersect_line_line_nd
    # 這裡直接呼叫：
    # return intersect_line_line_nd(L1, L2)
    
    # 由於我們沒有實際的 ND 模組，這裡直接使用簡化邏輯 (與 ND 中交點邏輯一致)
    # 此處應直接依賴 ND 的 intersect_line_line_nd 函式，但為確保範例執行，
    # 假設它能處理 3D 異面情況
    return intersect_line_line_nd(L1, L2)

def intersect_line_sphere_3d(L: Line, S: Sphere) -> List[PointND]:
    """
    【簡化/別名】計算直線與球體的交點。
    直接使用 ND 版本。
    """
    return intersect_line_sphere(L, S)

# --- 4. 3D 專屬轉換 (基於 ND 矩陣邏輯的簡化介面) ---

class Transformations3D:
    """提供 3D 空間的變換操作"""
    
    @staticmethod
    def translate(obj, T: VectorND):
        """【使用 ND】直接呼叫 ND 的平移 (假設 ND 中也有 translate 函式)"""
        # 由於沒有 ND 中的 translate，我們自行實作，但邏輯與 ND 相同
        if isinstance(obj, Line):
            return Line(obj.p + T, obj.direction)
        elif isinstance(obj, Sphere):
            return Sphere(obj.center + T, obj.radius)
        elif isinstance(obj, np.ndarray):
            return obj + T
        return obj

    @staticmethod
    def _rotate_on_axis(obj, axis: str, angle_deg: float):
        """
        核心旋轉函式：繞單一座標軸 (X, Y, Z) 旋轉。
        使用 TransformationsND._get_rotation_matrix 產生 3x3 矩陣。
        """
        angle_rad = math.radians(angle_deg)
        axis = axis.upper()
        
        # 3D 旋轉矩陣生成：
        # R(X): 繞 X 軸 (0) 旋轉，發生在 YZ (1-2) 平面
        # R(Y): 繞 Y 軸 (1) 旋轉，發生在 XZ (0-2) 平面
        # R(Z): 繞 Z 軸 (2) 旋轉，發生在 XY (0-1) 平面

        if axis == 'X':
            R_matrix = TransformationsND._get_rotation_matrix(3, 1, 2, angle_rad)
        elif axis == 'Y':
            R_matrix = TransformationsND._get_rotation_matrix(3, 0, 2, angle_rad)
        elif axis == 'Z':
            R_matrix = TransformationsND._get_rotation_matrix(3, 0, 1, angle_rad)
        else:
            raise ValueError("Axis must be 'X', 'Y', or 'Z'.")
            
        # 呼叫 ND 邏輯進行點旋轉
        def rotate_point_nd(P: PointND, R: np.ndarray) -> PointND:
            return R @ P

        if isinstance(obj, Line):
            return Line(
                rotate_point_nd(obj.p, R_matrix), 
                rotate_point_nd(obj.direction, R_matrix)
            )
        elif isinstance(obj, Sphere):
            return Sphere(
                rotate_point_nd(obj.center, R_matrix), 
                obj.radius
            )
        elif isinstance(obj, np.ndarray):
            return rotate_point_nd(obj, R_matrix)
        return obj

    @staticmethod
    def rotate_x(obj, angle_deg: float):
        """繞 X 軸旋轉"""
        return Transformations3D._rotate_on_axis(obj, 'X', angle_deg)
        
    @staticmethod
    def rotate_y(obj, angle_deg: float):
        """繞 Y 軸旋轉"""
        return Transformations3D._rotate_on_axis(obj, 'Y', angle_deg)
        
    @staticmethod
    def rotate_z(obj, angle_deg: float):
        """繞 Z 軸旋轉"""
        return Transformations3D._rotate_on_axis(obj, 'Z', angle_deg)


# --- 範例執行 (3D) ---
if __name__ == "__main__":
    
    # 範例點 (3D)
    P_A = np.array([1.0, 1.0, 1.0])
    V1 = np.array([1.0, 0.0, 0.0])
    V2 = np.array([0.0, 1.0, 0.0])
    
    # 範例直線
    Line_X = Line(p=np.array([0.0, 0.0, 0.0]), direction=V1) # X 軸
    
    # 範例球體
    Sphere1 = Sphere(center=np.array([5.0, 0.0, 0.0]), radius=3.0)

    print("--- 3D 幾何套件 (基於 ND 核心) ---")

    # 1. 3D 叉積 (3D 專屬)
    cp = cross_product_3d(V1, V2) # i x j = k
    print(f"1. 3D 叉積 V1 x V2: {cp}") # 應為 [0., 0., 1.]

    # 2. 直線與球體交點 (使用 ND 基礎)
    # L: X 軸, S: center=[5, 0, 0], R=3
    # 垂足為 [5, 0, 0]，距離為 0。應有兩個交點：[5-3, 0, 0] 和 [5+3, 0, 0]
    intersection_LS = intersect_line_sphere_3d(Line_X, Sphere1)
    print(f"2. 直線與球體交點: {np.array(intersection_LS).round(4)}") # 應為 [2., 0., 0.] 和 [8., 0., 0.]
    
    # 3. 旋轉操作 (3D 專屬介面)
    point_rotated_y = Transformations3D.rotate_y(P_A, 90) # 繞 Y 軸旋轉 90 度
    print(f"3. 點 {P_A} 繞 Y 軸旋轉 90 度: {point_rotated_y.round(4)}") # 應為 [1, 1, -1] -> [-1, 1, 1] 
    # (X->Z, Z->-X, Y不變) (1, 1, 1) -> (1, 1, -1) -> Z=1 -> X=-1
    # 實際結果: X -> Z, Z -> -X. (1, 1, 1) -> (1, 1, -1) [錯誤] -> 應為 (1, 1, -1) [Z=1 去了 X=1]
    # R(Y): [[c, 0, s], [0, 1, 0], [-s, 0, c]]. R(90): [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
    # [1, 1, 1] @ R: [1*0 + 1*0 + 1*(-1), 1*0 + 1*1 + 1*0, 1*1 + 1*0 + 1*0] = [-1, 1, 1]
    
    # 重新計算旋轉 (應為 [-1, 1, 1])
    print(f"3. 點 {P_A} 繞 Y 軸旋轉 90 度: {point_rotated_y.round(4)}")