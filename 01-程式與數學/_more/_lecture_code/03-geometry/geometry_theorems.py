import math
from geometry_objects import Point, EPSILON

def is_right_triangle(p1, p2, p3):
    """
    驗證由三個點 p1, p2, p3 所形成的三角形是否為直角三角形。
    
    參數:
        p1 (Point): 第一個頂點。
        p2 (Point): 第二個頂點。
        p3 (Point): 第三個頂點。
        
    回傳:
        bool: 如果是直角三角形，回傳 True；否則回傳 False。
    """
    # 檢查輸入是否為 Point 物件
    if not all(isinstance(p, Point) for p in [p1, p2, p3]):
        raise TypeError("All inputs must be Point objects.")

    # 檢查三個點是否共線
    # 三點共線則不能構成三角形。判斷方式是檢查任意兩點之間的方向向量是否平行。
    # 兩個向量 (x1, y1) 和 (x2, y2) 平行的條件是 x1*y2 - x2*y1 = 0
    v1 = p2 - p1
    v2 = p3 - p1
    if math.isclose(v1.x * v2.y - v1.y * v2.x, 0, abs_tol=EPSILON):
        # 也可以用斜率判斷，但垂直線的斜率是無限大，用向量叉積更通用
        # 或者簡單判斷任意兩點的距離和是否等於第三點距離
        if math.isclose(p1.distance_to(p2) + p2.distance_to(p3), p1.distance_to(p3), abs_tol=EPSILON) or \
           math.isclose(p1.distance_to(p3) + p3.distance_to(p2), p1.distance_to(p2), abs_tol=EPSILON) or \
           math.isclose(p2.distance_to(p1) + p1.distance_to(p3), p2.distance_to(p3), abs_tol=EPSILON):
            return False # 三點共線，不是三角形

    # 計算三條邊的長度
    a_sq = p1.distance_to(p2) ** 2
    b_sq = p2.distance_to(p3) ** 2
    c_sq = p3.distance_to(p1) ** 2
    
    # 將平方長度排序，找出最短兩邊和最長邊
    sides_sq = sorted([a_sq, b_sq, c_sq])
    
    # 驗證畢氏定理: 最短兩邊平方和是否等於最長邊平方
    return math.isclose(sides_sq[0] + sides_sq[1], sides_sq[2], abs_tol=EPSILON)

