# test_geometry_axioms.py
import pytest
import random
import math
from geometry_objects import Point, Line, ORIGIN, EPSILON

# 測試用的座標範圍
COORD_RANGE = 10
NUM_TEST_CASES = 50

# 輔助函式，生成隨機點
def get_random_point():
    return Point(random.uniform(-COORD_RANGE, COORD_RANGE),
                 random.uniform(-COORD_RANGE, COORD_RANGE))

# 輔助函式，生成兩個不同的隨機點
def get_two_distinct_random_points():
    p1 = get_random_point()
    p2 = get_random_point()
    while p1 == p2: # 確保兩點不同
        p2 = get_random_point()
    return p1, p2

# ----------------------------------------------------------------------
# 點和距離的法則
# ----------------------------------------------------------------------

# 1. 兩點之間有且只有一條直線 (由 Line 類別的構造隱含)
#    我們不能直接測試「有且只有一條」，但可以測試其屬性。

# 2. 距離的非負性
def test_distance_non_negativity():
    for _ in range(NUM_TEST_CASES):
        p1 = get_random_point()
        p2 = get_random_point()
        assert p1.distance_to(p2) >= 0, "Distance must be non-negative."

# 3. 距離的同一性 (Identity)
def test_distance_identity():
    for _ in range(NUM_TEST_CASES):
        p = get_random_point()
        assert math.isclose(p.distance_to(p), 0, abs_tol=EPSILON), "Distance from a point to itself must be zero."
        # 如果距離是0，則兩點必須相同 (反向驗證)
        p1, p2 = get_two_distinct_random_points()
        if math.isclose(p1.distance_to(p2), 0, abs_tol=EPSILON):
             assert p1 == p2, "If distance is zero, points must be identical."

# 4. 距離的對稱性 (Symmetry)
def test_distance_symmetry():
    for _ in range(NUM_TEST_CASES):
        p1 = get_random_point()
        p2 = get_random_point()
        assert math.isclose(p1.distance_to(p2), p2.distance_to(p1), abs_tol=EPSILON), \
            "Distance must be symmetric."

# 5. 距離的三角不等式 (Triangle Inequality)
def test_distance_triangle_inequality():
    for _ in range(NUM_TEST_CASES):
        p1 = get_random_point()
        p2 = get_random_point()
        p3 = get_random_point()
        # p1 到 p3 的距離 <= p1 到 p2 的距離 + p2 到 p3 的距離
        assert p1.distance_to(p3) <= p1.distance_to(p2) + p2.distance_to(p3) + EPSILON, \
            "Triangle inequality failed."

# ----------------------------------------------------------------------
# 直線的法則
# ----------------------------------------------------------------------

# 6. 點在直線上
def test_point_on_line():
    for _ in range(NUM_TEST_CASES):
        p1, p2 = get_two_distinct_random_points()
        line = Line(p1, p2)
        assert line.contains_point(p1), f"Line {line} should contain its defining point {p1}."
        assert line.contains_point(p2), f"Line {line} should contain its defining point {p2}."

        # 測試線上任意一點
        mid_point = Point((p1.x + p2.x) / 2, (p1.y + p2.y) / 2)
        assert line.contains_point(mid_point), f"Line {line} should contain its midpoint {mid_point}."

        # 測試線外一點 (隨機生成一個點，然後檢查它是否在線上)
        # 這裡需要確保生成的點不在線上，有點複雜，可以簡化為檢查一個明顯不在線上的點
        # 例如，如果線是水平的，則改變 y 座標的點就不在線上
        if not math.isclose(p1.y, p2.y, abs_tol=EPSILON): # 如果不是水平線
            outside_point = Point(p1.x, p1.y + 1)
            if not line.contains_point(outside_point): # 確保它真的不在線上
                 assert not line.contains_point(outside_point), f"Line {line} should not contain {outside_point}."
        else: # 如果是水平線
            outside_point = Point(p1.x + 1, p1.y + 1)
            if not line.contains_point(outside_point):
                 assert not line.contains_point(outside_point), f"Line {line} should not contain {outside_point}."


# 7. 兩條直線平行
def test_parallel_lines():
    for _ in range(NUM_TEST_CASES):
        p1, p2 = get_two_distinct_random_points()
        line1 = Line(p1, p2)

        # 構造一條與 line1 平行的線
        # 讓 line2 的方向向量與 line1 相同，但起點不同
        shift_vector = Point(random.uniform(1, 5), random.uniform(1, 5)) # 隨機位移
        p3 = p1 + shift_vector
        p4 = p2 + shift_vector
        line2 = Line(p3, p4)

        assert line1.is_parallel_to(line2), f"Line {line1} should be parallel to {line2}."

        # 測試一條不平行的線 (例如旋轉其中一個方向向量)
        # 簡單地改變一個點
        p5 = Point(p1.x + 1, p1.y + 2)
        p6 = Point(p2.x + 3, p2.y + 1)
        line3 = Line(p5, p6)
        # 只有在它們確實不平行時才斷言不平行
        if not line1.is_parallel_to(line3):
            assert not line1.is_parallel_to(line3), f"Line {line1} should not be parallel to {line3}."


# 8. 兩條直線垂直
def test_perpendicular_lines():
    for _ in range(NUM_TEST_CASES):
        p1, p2 = get_two_distinct_random_points()
        line1 = Line(p1, p2)

        # 構造一條與 line1 垂直的線
        # 如果 line1 的方向向量是 (dx, dy)，則垂直向量是 (-dy, dx) 或 (dy, -dx)
        dir_vec = line1.direction_vector
        perp_vec = Point(-dir_vec.y, dir_vec.x) # 垂直方向向量

        # 讓 line2 經過 p1，方向是 perp_vec
        p3 = p1 + perp_vec
        line2 = Line(p1, p3) # line2 經過 p1，方向垂直於 line1

        assert line1.is_perpendicular_to(line2), f"Line {line1} should be perpendicular to {line2}."

        # 測試一條不垂直的線 (例如平行線)
        p4 = Point(p1.x + 1, p1.y + 1)
        p5 = Point(p2.x + 1, p2.y + 1)
        line3 = Line(p4, p5) # line3 平行於 line1
        if not line1.is_perpendicular_to(line3): # 只有在確實不垂直時才斷言不垂直
            assert not line1.is_perpendicular_to(line3), f"Line {line1} should not be perpendicular to {line3}."

# ----------------------------------------------------------------------
# 面的法則 (在 2D 中，「面」就是整個平面，所以法則會比較簡單)
# 如果是 3D，則需要定義 Plane 類別，並測試共面性等。
# ----------------------------------------------------------------------

# 9. 三點共線 (Collinearity)
def test_collinearity():
    for _ in range(NUM_TEST_CASES):
        p1, p2 = get_two_distinct_random_points()
        line = Line(p1, p2)

        # 構造一個與 p1, p2 共線的點
        # p3 = p1 + k * (p2 - p1)
        k = random.uniform(-2, 2) # k 可以是任意實數
        p3 = p1 + (p2 - p1) * k
        assert line.contains_point(p3), f"Point {p3} should be collinear with {p1} and {p2}."

        # 構造一個不共線的點 (確保它真的不共線)
        p4 = Point(p1.x + 1, p1.y + 1)
        if not line.contains_point(p4):
            assert not line.contains_point(p4), f"Point {p4} should not be collinear with {p1} and {p2}."


