import math
from geometry_objects import Point, Line, ORIGIN, EPSILON

def intersect_lines(line1, line2):
    """
    計算兩條直線的交點。
    
    Args:
        line1 (Line): 第一條直線。
        line2 (Line): 第二條直線。
        
    Returns:
        Point: 如果兩條直線相交，則回傳交點；若平行或重合，則回傳 None。
    """
    if line1.is_parallel_to(line2):
        return None  # 如果兩條直線平行，則沒有唯一交點
    
    # 使用克拉瑪法則 (Cramer's Rule) 求解二元一次聯立方程式
    # line1: A1*x + B1*y = -C1
    # line2: A2*x + B2*y = -C2
    
    A1, B1, C1 = line1.A, line1.B, line1.C
    A2, B2, C2 = line2.A, line2.B, line2.C
    
    # 決定式 (Determinant)
    # D = A1*B2 - A2*B1
    D = A1 * B2 - A2 * B1
    
    # Dx = (-C1)*B2 - (-C2)*B1
    Dx = -C1 * B2 - (-C2) * B1
    
    # Dy = A1*(-C2) - A2*(-C1)
    Dy = A1 * (-C2) - A2 * (-C1)
    
    # 由於前面已經檢查過平行，D 不會為 0
    x = Dx / D
    y = Dy / D
    
    return Point(x, y)

def create_parallel_line(line, p):
    """
    給定一條直線和線外一點，創建一條通過該點且與原直線平行的直線。
    
    Args:
        line (Line): 原直線。
        p (Point): 線外一點。
        
    Returns:
        Line: 新創建的平行線。
    """
    if line.contains_point(p):
        # 點在直線上，無法創建唯一平行線
        # 這裡的處理方式是返回一條與原直線重合的線
        print("警告：該點已在直線上，返回一條與原直線重合的線。")
    
    # 使用原直線的方向向量，並從新點 p 出發創建一條新線
    # 只需要從 p 點沿著 direction_vector 找到另一個點即可
    p2 = p + line.direction_vector
    return Line(p, p2)

def find_perpendicular_foot(line, p):
    """
    從線外一點向直線做一條垂直線，並找出其交點（垂足）。
    
    Args:
        line (Line): 原直線。
        p (Point): 線外一點。
        
    Returns:
        Point: 從 p 點到直線的垂足。
    """
    # 如果點已經在直線上，垂足就是該點本身
    if line.contains_point(p):
        return p
    
    # 1. 找出直線的法向量 (Normal Vector)
    # 直線 Ax + By + C = 0，法向量為 (A, B)
    normal_vector = Point(line.A, line.B)
    
    # 2. 創建一條通過 p 點，且方向為法向量的直線（即垂直線）
    # 因為法向量與直線垂直，所以以法向量為方向的直線就會是我們的垂直線
    perpendicular_line = create_line_from_point_and_direction(p, normal_vector)
    
    # 3. 找出兩條直線的交點，即為垂足
    return intersect_lines(line, perpendicular_line)

def create_line_from_point_and_direction(p_start, direction_vector):
    """
    輔助函式：根據起點和方向向量創建一條直線。
    """
    if not isinstance(p_start, Point) or not isinstance(direction_vector, Point):
        raise TypeError("Input must be Point objects.")
    
    p_end = p_start + direction_vector
    return Line(p_start, p_end)


# --- 測試範例 ---
if __name__ == "__main__":
    # 測試兩直線交點
    print("--- 測試兩直線交點 ---")
    l1 = Line(Point(0, 0), Point(5, 5))  # y = x
    l2 = Line(Point(0, 5), Point(5, 0))  # y = -x + 5
    intersection = intersect_lines(l1, l2)
    print(f"l1: {l1}")
    print(f"l2: {l2}")
    print(f"交點: {intersection}")
    print(f"期望結果: Point(2.5, 2.5)\n")

    # 測試平行線
    l3 = Line(Point(1, 1), Point(2, 2))  # y = x
    l4 = Line(Point(0, 1), Point(1, 2))  # y = x + 1
    parallel_intersection = intersect_lines(l3, l4)
    print(f"l3: {l3}")
    print(f"l4: {l4}")
    print(f"交點 (應為 None): {parallel_intersection}\n")
    
    # 測試創建平行線
    print("--- 測試創建平行線 ---")
    line_to_parallel = Line(Point(0, 0), Point(1, 0))  # x 軸
    point_for_parallel = Point(5, 3)
    parallel_line = create_parallel_line(line_to_parallel, point_for_parallel)
    print(f"原直線: {line_to_parallel}")
    print(f"線外一點: {point_for_parallel}")
    print(f"新平行線: {parallel_line}")
    print(f"新線是否平行: {line_to_parallel.is_parallel_to(parallel_line)}")
    print(f"新線是否通過點: {parallel_line.contains_point(point_for_parallel)}\n")

    # 測試找出垂足
    print("--- 測試找出垂足 ---")
    line_for_foot = Line(Point(0, 0), Point(10, 0))  # x 軸
    point_for_foot = Point(5, 3)
    perpendicular_foot = find_perpendicular_foot(line_for_foot, point_for_foot)
    print(f"直線: {line_for_foot}")
    print(f"線外一點: {point_for_foot}")
    print(f"垂足: {perpendicular_foot}")
    print(f"期望結果: Point(5.0, 0.0)\n")

    # 測試垂足：點在線上
    line_for_foot_2 = Line(Point(0, 0), Point(10, 10))  # y = x
    point_on_line = Point(5, 5)
    perpendicular_foot_2 = find_perpendicular_foot(line_for_foot_2, point_on_line)
    print(f"直線: {line_for_foot_2}")
    print(f"線上一點: {point_on_line}")
    print(f"垂足: {perpendicular_foot_2}")
    print(f"期望結果: Point(5.0, 5.0)")