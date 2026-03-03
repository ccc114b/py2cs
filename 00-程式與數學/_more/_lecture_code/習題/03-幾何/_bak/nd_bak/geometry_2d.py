import numpy as np
import math
from typing import Union, List, Tuple

from geometry_nd import (
    VectorND, PointND, LineND, Sphere, 
    get_perpendicular_foot_nd, intersect_line_line_nd, intersect_line_sphere,
    TransformationsND
)

# --- 1. 2D 幾何物件定義 (繼承/別名) ---

# 將 ND 的類別名稱改成 2D 中常用的別名，並強制執行 D=2
class Line(LineND):
    """2D 直線：基於 LineND，強制維度 D=2"""
    def __init__(self, p: PointND, direction: VectorND):
        if p.shape[0] != 2 or direction.shape[0] != 2:
            raise ValueError("Line must be initialized with 2D vectors.")
        super().__init__(p, direction)

class Circle(Sphere):
    """2D 圓：基於 Sphere，強制維度 D=2"""
    def __init__(self, center: PointND, radius: float):
        if center.shape[0] != 2:
            raise ValueError("Circle center must be a 2D vector.")
        super().__init__(center, radius)

# --- 2. 2D 專屬向量運算 ---

def cross_product_2d(v1: VectorND, v2: VectorND) -> float:
    """【2D 專屬】計算 2D 向量的叉積（返回 Z 分量，純量）"""
    return v1[0] * v2[1] - v1[1] * v2[0]

# --- 3. 2D 專屬相交運算 (使用 ND 基礎) ---

def intersect_line_line(L1: Line, L2: Line) -> Union[PointND, None]:
    """
    【簡化/別名】計算兩直線交點。
    直接使用 ND 版本，但因強制 D=2，其行為符合 2D 線性方程組求解。
    """
    return intersect_line_line_nd(L1, L2)

def intersect_line_circle(L: Line, C: Circle) -> List[PointND]:
    """
    【簡化/別名】計算直線與圓的交點。
    直接使用 ND 版本，因為線-球體交點邏輯在 D=2 時即為線-圓交點。
    """
    return intersect_line_sphere(L, C)

def intersect_circle_circle(C1: Circle, C2: Circle) -> List[PointND]:
    """【2D 專屬】計算兩圓 C1 和 C2 的交點 (ND 中因 N>3 複雜而被移除)。"""
    d = np.linalg.norm(C2.center - C1.center)
    r1, r2 = C1.radius, C2.radius
    
    if d > r1 + r2 or d < abs(r1 - r2) or np.isclose(d, 0):
        return []

    # 計算交點到圓心連線上的距離 a (與 ND 的 intersect_sphere_sphere 邏輯一致)
    a = (r1**2 - r2**2 + d**2) / (2 * d)
    a = np.clip(a, -d, d)
    
    # 交點線段到圓心連線的距離 h
    h_sq = r1**2 - a**2
    h = math.sqrt(max(0, h_sq))
    
    # 圓心連線的方向單位向量 e12
    e12 = (C2.center - C1.center) / d
    
    # 交點線段的中點 P_c
    P_c = C1.center + a * e12
    
    if np.isclose(h, 0):
        return [P_c] # 相切
        
    # 垂直於 e12 的向量 (2D 專屬)
    perp_e = np.array([-e12[1], e12[0]])
    
    I1 = P_c + h * perp_e
    I2 = P_c - h * perp_e
    
    # 考慮重合 (誤差導致的兩個極近的點)
    if np.linalg.norm(I1 - I2) < 1e-9:
         return [I1]
         
    return [I1, I2]

# --- 4. 2D 變換 (基於 ND 架構的簡化) ---

class Transformations2D:
    """提供 2D 空間的變換操作"""
    
    @staticmethod
    def translate(obj, T: VectorND):
        """【使用 ND】直接呼叫 ND 的平移 (假設 ND 中也有 translate 函式)"""
        if isinstance(obj, Line):
            return Line(obj.p + T, obj.direction)
        elif isinstance(obj, Circle):
            return Circle(obj.center + T, obj.radius)
        elif isinstance(obj, np.ndarray):
            return obj + T
        return obj

    @staticmethod
    def rotate(obj, angle_deg: float):
        """
        【2D 專屬介面】對物件進行旋轉 (角度，逆時針)。
        利用 TransformationsND 的核心矩陣生成邏輯。
        """
        angle_rad = math.radians(angle_deg)
        
        # 2D 旋轉即為繞 Z 軸旋轉，發生在 XY (0-1) 平面
        # 這裡直接使用 2D 旋轉矩陣
        c = math.cos(angle_rad)
        s = math.sin(angle_rad)
        R_matrix = np.array([[c, -s], [s, c]])
        
        # 定義一個內部輔助函式，處理 PointND 的旋轉
        def rotate_point_nd(P: PointND, R: np.ndarray) -> PointND:
            return R @ P

        if isinstance(obj, Line):
            return Line(
                rotate_point_nd(obj.p, R_matrix), 
                rotate_point_nd(obj.direction, R_matrix)
            )
        elif isinstance(obj, Circle):
            return Circle(
                rotate_point_nd(obj.center, R_matrix), 
                obj.radius
            )
        elif isinstance(obj, np.ndarray):
            return rotate_point_nd(obj, R_matrix)
        return obj

# --- 範例執行 (2D) ---
if __name__ == "__main__":
    
    # 範例點 (2D)
    P_A = np.array([1.0, 1.0])
    
    # 範例直線
    Line1 = Line(p=np.array([1.0, 1.0]), direction=np.array([4.0, 4.0]))
    
    # 範例圓
    Circle1 = Circle(center=np.array([4.0, 1.0]), radius=3.0)
    Circle2 = Circle(center=np.array([8.0, 1.0]), radius=5.0)

    print("--- 2D 幾何套件 (基於 ND 核心) ---")

    # 1. 兩直線交點 (使用 ND 基礎)
    Line_H = Line(p=np.array([0.0, 1.0]), direction=np.array([1.0, 0.0])) # y=1
    Line_V = Line(p=np.array([4.0, 0.0]), direction=np.array([0.0, 1.0])) # x=4
    intersection_LL = intersect_line_line(Line_H, Line_V)
    print(f"1. 兩直線交點 (x=4, y=1): {intersection_LL}") # 應為 [4., 1.]

    # 2. 直線與圓交點 (使用 ND 基礎)
    intersection_LC = intersect_line_circle(Line_H, Circle1)
    print(f"2. 直線與圓交點 (y=1, C1): {np.array(intersection_LC).round(4)}") # 應為 [1., 1.] 和 [7., 1.]

    # 3. 兩圓交點 (2D 專屬)
    intersection_CC = intersect_circle_circle(Circle1, Circle2)
    print(f"3. 兩圓交點 (C1, C2): {np.array(intersection_CC).round(4)}") # 應為 (4, 4) 和 (4, -2)

    # 4. 旋轉操作 (2D 專屬介面)
    point_rotated = Transformations2D.rotate(P_A, 90)
    print(f"4. 點 {P_A} 旋轉 90 度: {point_rotated.round(4)}") # 應為 [-1., 1.]