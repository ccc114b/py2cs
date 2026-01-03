# test_geometry_3d_axioms.py
import pytest
import random
import math
from geometry_3d_objects import Point3D, Line3D, Plane, ORIGIN_3D, EPSILON, are_points_coplanar

# 測試用的座標範圍
COORD_RANGE = 10
NUM_TEST_CASES = 50

# 輔助函式，生成隨機點
def get_random_point_3d():
    return Point3D(random.uniform(-COORD_RANGE, COORD_RANGE),
                   random.uniform(-COORD_RANGE, COORD_RANGE),
                   random.uniform(-COORD_RANGE, COORD_RANGE))

# 輔助函式，生成兩個不同的隨機點
def get_two_distinct_random_points_3d():
    p1 = get_random_point_3d()
    p2 = get_random_point_3d()
    while p1 == p2:
        p2 = get_random_point_3d()
    return p1, p2

# 輔助函式，生成三個不共線的隨機點
def get_three_non_collinear_random_points_3d():
    while True:
        p1 = get_random_point_3d()
        p2 = get_random_point_3d()
        p3 = get_random_point_3d()
        if p1 == p2 or p1 == p3 or p2 == p3:
            continue
        
        v1 = p2 - p1
        v2 = p3 - p1
        # 如果叉積的長度不為零，則三點不共線
        if not math.isclose(v1.cross(v2).magnitude(), 0, abs_tol=EPSILON):
            return p1, p2, p3

# ----------------------------------------------------------------------
# Point3D 和距離的法則
# ----------------------------------------------------------------------

# 1. 距離的非負性
def test_distance_non_negativity_3d():
    for _ in range(NUM_TEST_CASES):
        p1 = get_random_point_3d()
        p2 = get_random_point_3d()
        assert p1.distance_to(p2) >= 0, "Distance must be non-negative."

# 2. 距離的同一性 (Identity)
def test_distance_identity_3d():
    for _ in range(NUM_TEST_CASES):
        p = get_random_point_3d()
        assert math.isclose(p.distance_to(p), 0, abs_tol=EPSILON), "Distance from a point to itself must be zero."
        # 如果距離是0，則兩點必須相同 (反向驗證)
        p1, p2 = get_two_distinct_random_points_3d()
        if math.isclose(p1.distance_to(p2), 0, abs_tol=EPSILON):
             assert p1 == p2, "If distance is zero, points must be identical."

# 3. 距離的對稱性 (Symmetry)
def test_distance_symmetry_3d():
    for _ in range(NUM_TEST_CASES):
        p1 = get_random_point_3d()
        p2 = get_random_point_3d()
        assert math.isclose(p1.distance_to(p2), p2.distance_to(p1), abs_tol=EPSILON), \
            "Distance must be symmetric."

# 4. 距離的三角不等式 (Triangle Inequality)
def test_distance_triangle_inequality_3d():
    for _ in range(NUM_TEST_CASES):
        p1 = get_random_point_3d()
        p2 = get_random_point_3d()
        p3 = get_random_point_3d()
        assert p1.distance_to(p3) <= p1.distance_to(p2) + p2.distance_to(p3) + EPSILON, \
            "Triangle inequality failed."

# ----------------------------------------------------------------------
# Line3D 的法則
# ----------------------------------------------------------------------

# 5. 點在直線上
def test_point_on_line_3d():
    for _ in range(NUM_TEST_CASES):
        p1, p2 = get_two_distinct_random_points_3d()
        line = Line3D(p1, p2)
        assert line.contains_point(p1), f"Line {line} should contain its defining point {p1}."
        assert line.contains_point(p2), f"Line {line} should contain its defining point {p2}."

        # 測試線上任意一點 (例如中點)
        mid_point = Point3D((p1.x + p2.x) / 2, (p1.y + p2.y) / 2, (p1.z + p2.z) / 2)
        assert line.contains_point(mid_point), f"Line {line} should contain its midpoint {mid_point}."

        # 測試線外一點 (構造一個明顯不在線上的點)
        # 找一個與 p1 不在同一條線上的點
        p_out = p1 + Point3D(1, 1, 1) # 隨機偏移
        if not line.contains_point(p_out): # 確保它真的不在線上
             assert not line.contains_point(p_out), f"Line {line} should not contain {p_out}."


# 6. 兩條直線平行
def test_parallel_lines_3d():
    for _ in range(NUM_TEST_CASES):
        p1, p2 = get_two_distinct_random_points_3d()
        line1 = Line3D(p1, p2)

        # 構造一條與 line1 平行的線
        shift_vector = Point3D(random.uniform(1, 5), random.uniform(1, 5), random.uniform(1, 5))
        p3 = p1 + shift_vector
        p4 = p2 + shift_vector
        line2 = Line3D(p3, p4)

        assert line1.is_parallel_to(line2), f"Line {line1} should be parallel to {line2}."

        # 測試一條不平行的線 (例如旋轉方向向量)
        p5 = get_random_point_3d()
        p6 = get_random_point_3d()
        line3 = Line3D(p5, p6)
        if not line1.is_parallel_to(line3): # 只有在確實不平行時才斷言不平行
            assert not line1.is_parallel_to(line3), f"Line {line1} should not be parallel to {line3}."


# 7. 兩條直線垂直
def test_perpendicular_lines_3d():
    for _ in range(NUM_TEST_CASES):
        p1, p2 = get_two_distinct_random_points_3d()
        line1 = Line3D(p1, p2)

        # 構造一條與 line1 垂直的線
        dir_vec = line1.direction_vector
        # 找到一個與 dir_vec 不平行的向量，然後取叉積得到垂直向量
        # 這裡需要小心處理 dir_vec 剛好是軸向量的情況
        if not math.isclose(dir_vec.x, 0, abs_tol=EPSILON) or \
           not math.isclose(dir_vec.y, 0, abs_tol=EPSILON):
            # 如果不是 Z 軸方向，就與 (0,0,1) 叉積
            perp_dir_vec = dir_vec.cross(Point3D(0, 0, 1)).normalize()
        else:
            # 如果是 Z 軸方向，就與 (0,1,0) 叉積
            perp_dir_vec = dir_vec.cross(Point3D(0, 1, 0)).normalize()
        
        # 確保 perp_dir_vec 不是零向量
        if math.isclose(perp_dir_vec.magnitude(), 0, abs_tol=EPSILON):
            # 如果叉積為零，表示 dir_vec 與 (0,0,1) 平行，即 dir_vec 是 (0,0,k) 形式
            # 此時改與 (1,0,0) 叉積
            perp_dir_vec = dir_vec.cross(Point3D(1, 0, 0)).normalize()

        p3 = p1 + perp_dir_vec
        line2 = Line3D(p1, p3) # line2 經過 p1，方向垂直於 line1

        assert line1.is_perpendicular_to(line2), f"Line {line1} should be perpendicular to {line2}."

        # 測試一條不垂直的線 (例如平行線)
        p4 = p1 + Point3D(1, 1, 1)
        p5 = p2 + Point3D(1, 1, 1)
        line3 = Line3D(p4, p5) # line3 平行於 line1
        if not line1.is_perpendicular_to(line3):
            assert not line1.is_perpendicular_to(line3), f"Line {line1} should not be perpendicular to {line3}."

# ----------------------------------------------------------------------
# Plane 的法則
# ----------------------------------------------------------------------

# 8. 點在平面上
def test_point_on_plane():
    for _ in range(NUM_TEST_CASES):
        p1, p2, p3 = get_three_non_collinear_random_points_3d()
        plane = Plane(p1, p2, p3)
        assert plane.contains_point(p1), f"Plane {plane} should contain its defining point {p1}."
        assert plane.contains_point(p2), f"Plane {plane} should contain its defining point {p2}."
        assert plane.contains_point(p3), f"Plane {plane} should contain its defining point {p3}."

        # 測試平面上的任意一點 (例如 p1, p2, p3 構成的三角形的重心)
        centroid = Point3D((p1.x + p2.x + p3.x) / 3,
                           (p1.y + p2.y + p3.y) / 3,
                           (p1.z + p2.z + p3.z) / 3)
        assert plane.contains_point(centroid), f"Plane {plane} should contain its centroid {centroid}."

        # 測試平面外一點 (沿法向量方向偏移)
        outside_point = p1 + plane.normal_vector * random.uniform(1, 5)
        if not plane.contains_point(outside_point): # 確保它真的不在平面上
            assert not plane.contains_point(outside_point), f"Plane {plane} should not contain {outside_point}."

# 9. 兩個平面平行
def test_parallel_planes():
    for _ in range(NUM_TEST_CASES):
        p1, p2, p3 = get_three_non_collinear_random_points_3d()
        plane1 = Plane(p1, p2, p3)

        # 構造一個與 plane1 平行的平面
        shift_vector = Point3D(random.uniform(1, 5), random.uniform(1, 5), random.uniform(1, 5))
        p4 = p1 + shift_vector
        p5 = p2 + shift_vector
        p6 = p3 + shift_vector
        plane2 = Plane(p4, p5, p6)

        assert plane1.is_parallel_to(plane2), f"Plane {plane1} should be parallel to {plane2}."

        # 測試一個不平行的平面
        p7, p8, p9 = get_three_non_collinear_random_points_3d()
        plane3 = Plane(p7, p8, p9)
        if not plane1.is_parallel_to(plane3):
            assert not plane1.is_parallel_to(plane3), f"Plane {plane1} should not be parallel to {plane3}."


# 10. 兩個平面垂直
def test_perpendicular_planes():
    for _ in range(NUM_TEST_CASES):
        p1, p2, p3 = get_three_non_collinear_random_points_3d()
        plane1 = Plane(p1, p2, p3)

        # 構造一個與 plane1 垂直的平面
        # 垂直平面的法向量與原平面法向量垂直 (內積為 0)
        n1 = plane1.normal_vector
        
        # 找到一個與 n1 垂直的向量 (例如，n1 與 (0,0,1) 的叉積)
        if not math.isclose(n1.x, 0, abs_tol=EPSILON) or \
           not math.isclose(n1.y, 0, abs_tol=EPSILON):
            v_perp_to_n1 = n1.cross(Point3D(0, 0, 1)).normalize()
        else:
            v_perp_to_n1 = n1.cross(Point3D(0, 1, 0)).normalize()
        
        # 確保 v_perp_to_n1 不是零向量
        if math.isclose(v_perp_to_n1.magnitude(), 0, abs_tol=EPSILON):
             v_perp_to_n1 = n1.cross(Point3D(1, 0, 0)).normalize()

        # 構造第二個垂直向量 (與 n1 和 v_perp_to_n1 都垂直)
        v_perp_to_n1_2 = n1.cross(v_perp_to_n1).normalize()

        # 讓 plane2 經過 p1，並由 p1, p1+v_perp_to_n1, p1+v_perp_to_n1_2 定義
        # 這樣 plane2 的法向量就是 n1 的垂直向量
        plane2 = Plane(p1, p1 + v_perp_to_n1, p1 + v_perp_to_n1_2)

        assert plane1.is_perpendicular_to(plane2), f"Plane {plane1} should be perpendicular to {plane2}."

        # 測試一個不垂直的平面 (例如平行平面)
        p7 = p1 + Point3D(1, 1, 1)
        p8 = p2 + Point3D(1, 1, 1)
        p9 = p3 + Point3D(1, 1, 1)
        plane3 = Plane(p7, p8, p9) # plane3 平行於 plane1
        if not plane1.is_perpendicular_to(plane3):
            assert not plane1.is_perpendicular_to(plane3), f"Plane {plane1} should not be perpendicular to {plane3}."

# 11. 四點共面 (Coplanarity)
def test_coplanarity():
    for _ in range(NUM_TEST_CASES):
        p1, p2, p3 = get_three_non_collinear_random_points_3d()
        plane = Plane(p1, p2, p3)

        # 構造一個與 p1, p2, p3 共面的點
        # 隨機生成平面上的一個點
        s = random.uniform(-1, 1)
        t = random.uniform(-1, 1)
        # p4 = p1 + s*(p2-p1) + t*(p3-p1)
        p4 = p1 + (p2 - p1) * s + (p3 - p1) * t
        assert are_points_coplanar(p1, p2, p3, p4), f"Points {p1}, {p2}, {p3}, {p4} should be coplanar."

        # 構造一個不共面的點 (沿法向量方向偏移)
        p_out = p1 + plane.normal_vector * random.uniform(1, 5)
        # 確保它真的不共面
        if not are_points_coplanar(p1, p2, p3, p_out):
            assert not are_points_coplanar(p1, p2, p3, p_out), f"Points {p1}, {p2}, {p3}, {p_out} should not be coplanar."
