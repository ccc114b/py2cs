# test_geometry_objects.py

import pytest
import math
from geometry_objects import Point, Line, Circle, ORIGIN, EPSILON

# ------------------------------------------------------------------
# Test Point Class
# ------------------------------------------------------------------

def test_point_init():
    """測試 Point 類別的初始化"""
    p = Point(1, 2)
    assert p.x == 1.0
    assert p.y == 2.0
    with pytest.raises(TypeError):
        Point("a", 2)
    with pytest.raises(TypeError):
        Point(1, "b")

def test_point_repr():
    """測試 __repr__ 方法"""
    p = Point(1, 2)
    assert repr(p) == "Point(1.0, 2.0)"

def test_point_equality():
    """測試點的相等性 (==) 和不相等性 (!=)"""
    p1 = Point(1, 2)
    p2 = Point(1.0, 2.0)
    p3 = Point(1.0 + EPSILON / 2, 2.0 - EPSILON / 2)
    p4 = Point(1.1, 2.2)
    assert p1 == p2
    assert p1 == p3
    assert p1 != p4
    assert p1 != "not a point"

def test_point_hash():
    """測試 __hash__ 方法"""
    p1 = Point(1, 2)
    p2 = Point(1.0, 2.0)
    d = {p1: "a"}
    assert d[p2] == "a"
    s = {p1, p2}
    assert len(s) == 1

def test_point_arithmetic():
    """測試點的加、減、乘法運算"""
    p1 = Point(3, 4)
    p2 = Point(1, 1)

    # 加法
    p_add = p1 + p2
    assert p_add == Point(4, 5)
    with pytest.raises(TypeError):
        _ = p1 + 5

    # 減法
    p_sub = p1 - p2
    assert p_sub == Point(2, 3)
    # 點 - 點 = 向量，向量的長度應該等於點到點的距離
    assert (p1 - p2).magnitude() == p1.distance_to(p2)
    with pytest.raises(TypeError):
        _ = p1 - 5

    # 純量乘法
    p_mul = p1 * 2
    assert p_mul == Point(6, 8)
    p_rmul = 2 * p1
    assert p_rmul == Point(6, 8)
    with pytest.raises(TypeError):
        _ = p1 * p2

def test_point_dot_magnitude_distance():
    """測試內積、長度與距離方法"""
    p1 = Point(3, 4) # 經典的勾股數向量
    p2 = Point(1, 1)
    p3 = Point(-4, 3)

    # 內積
    assert math.isclose(p1.dot(p2), 7)
    assert math.isclose(p1.dot(p3), 0) # 垂直向量內積為 0
    with pytest.raises(TypeError):
        p1.dot(5)

    # 長度 (magnitude)
    assert math.isclose(p1.magnitude(), 5)
    assert math.isclose(ORIGIN.magnitude(), 0)

    # 距離 (distance_to)
    assert math.isclose(p1.distance_to(ORIGIN), 5)
    assert math.isclose(p1.distance_to(p2), math.sqrt((3-1)**2 + (4-1)**2))
    with pytest.raises(TypeError):
        p1.distance_to(5)

def test_point_normalize():
    """測試單位向量方法"""
    p = Point(3, 4)
    unit_p = p.normalize()
    assert math.isclose(unit_p.magnitude(), 1)
    assert math.isclose(unit_p.x, 3/5)
    assert math.isclose(unit_p.y, 4/5)
    # 零向量
    zero_p = ORIGIN.normalize()
    assert zero_p == ORIGIN

# ------------------------------------------------------------------
# Test Line Class
# ------------------------------------------------------------------

def test_line_init():
    """測試 Line 類別的初始化"""
    p1 = Point(0, 0)
    p2 = Point(1, 1)
    line = Line(p1, p2)
    assert line.p1 == p1
    assert line.p2 == p2
    assert line.direction_vector == Point(1/math.sqrt(2), 1/math.sqrt(2))
    assert math.isclose(line.A, 1)
    assert math.isclose(line.B, -1)
    assert math.isclose(line.C, 0)

    with pytest.raises(TypeError):
        Line(p1, "not a point")
    with pytest.raises(ValueError):
        Line(p1, p1)

def test_line_contains_point():
    """測試 contains_point 方法"""
    line = Line(Point(0, 0), Point(2, 2))
    assert line.contains_point(Point(1, 1))
    assert not line.contains_point(Point(1, 2))
    # 浮點數誤差
    assert line.contains_point(Point(1 + EPSILON/2, 1 + EPSILON/2))
    with pytest.raises(TypeError):
        line.contains_point("not a point")

def test_line_parallel_perpendicular():
    """測試平行與垂直方法"""
    line1 = Line(Point(0, 0), Point(1, 1))
    line2 = Line(Point(0, 1), Point(1, 2)) # 平行
    line3 = Line(Point(0, 0), Point(-1, 1)) # 垂直
    line4 = Line(Point(0, 0), Point(1, 0)) # 水平線

    # 平行
    assert line1.is_parallel_to(line2)
    assert not line1.is_parallel_to(line3)
    assert not line1.is_parallel_to(line4)
    with pytest.raises(TypeError):
        line1.is_parallel_to("not a line")

    # 垂直
    assert line1.is_perpendicular_to(line3)
    assert not line1.is_perpendicular_to(line2)
    # assert line3.is_perpendicular_to(line4) # y=-x 與 x軸垂直 => AI 錯！
    with pytest.raises(TypeError):
        line1.is_perpendicular_to("not a line")

# ------------------------------------------------------------------
# Test Circle Class
# ------------------------------------------------------------------

def test_circle_init():
    """測試 Circle 類別的初始化"""
    c = Circle(Point(0, 0), 5)
    assert c.center == ORIGIN
    assert c.radius == 5.0
    assert c.radius_sq == 25.0

    with pytest.raises(TypeError):
        Circle("not a point", 5)
    with pytest.raises(TypeError):
        Circle(ORIGIN, "not a number")
    with pytest.raises(ValueError):
        Circle(ORIGIN, 0)
    with pytest.raises(ValueError):
        Circle(ORIGIN, -1)

def test_circle_repr_equality():
    """測試 __repr__ 和相等性"""
    c1 = Circle(Point(1, 2), 3)
    c2 = Circle(Point(1.0, 2.0), 3.0)
    c3 = Circle(Point(1, 2), 3.1)
    assert repr(c1) == "Circle(Point(1.0, 2.0), radius=3.0)"
    assert c1 == c2
    assert c1 != c3
    assert c1 != "not a circle"

def test_circle_area_circumference():
    """測試面積與周長"""
    c = Circle(Point(0, 0), 2)
    assert math.isclose(c.area(), math.pi * 4)
    assert math.isclose(c.circumference(), 2 * math.pi * 2)

@pytest.mark.parametrize("p, expected", [
    (Point(3, 0), True),  # 在圓周上
    (Point(0, 0), True),  # 在圓心
    (Point(1, 1), True),  # 在圓內
    (Point(4, 0), False), # 在圓外
    (Point(3 + EPSILON, 0), False), # 圓周外一點
    (Point(3 - EPSILON, 0), True) # 圓周內一點
])
def test_circle_contains_point(p, expected):
    """測試 contains_point 方法 (參數化測試)"""
    c = Circle(Point(0, 0), 3)
    assert c.contains_point(p) == expected

def test_circle_contains_point_type_error():
    """測試 contains_point 的類型錯誤"""
    c = Circle(Point(0, 0), 3)
    with pytest.raises(TypeError):
        c.contains_point("not a point")

@pytest.mark.parametrize("p1, p2, expected", [
    (Point(0, 5), Point(10, 5), True),  # 直線與圓心同 y 軸，相交
    (Point(5, 0), Point(5, 10), True),  # 直線與圓心同 x 軸，相交
    (Point(0, 6), Point(10, 6), False), # 直線在圓外，不相交
    # (Point(5, 5), Point(5, -5), False), # 直線經過圓心，但在 x 軸上，相交
    (Point(5, 5), Point(5, -5), True), # 直線經過圓心，但在 x 軸上，相交
    (Point(5, 0), Point(-5, 0), True), # 直線經過圓心，相交
])
def test_circle_intersects_line(p1, p2, expected):
    """測試圓與直線相交 (參數化測試)"""
    c = Circle(Point(0, 0), 5)
    line = Line(p1, p2)
    assert c.intersects_line(line) == expected
    # 測試相切的情況
    tangent_line = Line(Point(5, 0), Point(5, 1))
    assert c.intersects_line(tangent_line) is True

def test_circle_intersects_line_type_error():
    """測試 intersects_line 的類型錯誤"""
    c = Circle(Point(0, 0), 5)
    with pytest.raises(TypeError):
        c.intersects_line("not a line")

@pytest.mark.parametrize("center, radius, expected", [
    (Point(10, 0), 5, True),   # 兩圓外切
    (Point(0, 0), 2, False),    # 一圓在另一圓內，不相交
    (Point(0, 0), 10, True),   # 兩圓重疊
    (Point(20, 0), 4, False),  # 兩圓相離
    (Point(0, 0), 1, False),   # 一圓在另一圓內，不相交
    (Point(15, 0), 5, True),   # 兩圓相交
])
def test_circle_intersects_circle(center, radius, expected):
    """測試圓與圓相交 (參數化測試)"""
    c1 = Circle(Point(0, 0), 10)
    c2 = Circle(center, radius)
    assert c1.intersects_circle(c2) == expected

def test_circle_intersects_circle_type_error():
    """測試 intersects_circle 的類型錯誤"""
    c = Circle(Point(0, 0), 5)
    with pytest.raises(TypeError):
        c.intersects_circle("not a circle")