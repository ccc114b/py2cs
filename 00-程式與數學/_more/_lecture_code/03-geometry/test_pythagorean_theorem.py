import pytest
from geometry_objects import Point
from geometry_theorems import *
# from triangle_validator import is_right_triangle

# ------------------------------------------------------------------
# Test Cases for is_right_triangle function
# ------------------------------------------------------------------

def test_classic_right_triangle():
    """驗證經典的直角三角形 (3-4-5)"""
    p1 = Point(0, 0)
    p2 = Point(3, 0)
    p3 = Point(0, 4)
    assert is_right_triangle(p1, p2, p3)

def test_rotated_right_triangle():
    """驗證旋轉後的直角三角形"""
    p1 = Point(1, 1)
    p2 = Point(5, 1)
    p3 = Point(1, 4)
    assert is_right_triangle(p1, p2, p3)

def test_right_triangle_with_negative_coords():
    """驗證坐標為負數的直角三角形"""
    p1 = Point(-3, -4)
    p2 = Point(-3, 1)
    p3 = Point(2, -4)
    assert is_right_triangle(p1, p2, p3)

def test_non_right_triangle():
    """驗證非直角三角形"""
    p1 = Point(0, 0)
    p2 = Point(3, 0)
    p3 = Point(1, 4)  # 銳角三角形
    assert not is_right_triangle(p1, p2, p3)
    
def test_non_right_triangle_long_side():
    """驗證不符合畢氏定理的鈍角三角形"""
    p1 = Point(0, 0)
    p2 = Point(3, 0)
    p3 = Point(1, 5)
    assert not is_right_triangle(p1, p2, p3)


# --- 參數化測試 ---
# 這種寫法可以避免重複的測試程式碼，非常推薦
@pytest.mark.parametrize("p1, p2, p3", [
    (Point(0, 0), Point(5, 0), Point(0, 12)),  # 5-12-13 畢氏數
    (Point(0, 0), Point(6, 8), Point(-8, 6)),  # 垂直向量
    (Point(1.5, 2), Point(4.5, 2), Point(1.5, 6)), # 浮點數坐標
])
def test_parametrized_right_triangles(p1, p2, p3):
    """使用參數化驗證多個直角三角形案例"""
    assert is_right_triangle(p1, p2, p3)


def test_collinear_points():
    """驗證三點共線的情況，這不構成三角形"""
    p1 = Point(0, 0)
    p2 = Point(1, 1)
    p3 = Point(2, 2)
    assert not is_right_triangle(p1, p2, p3)

def test_collinear_points_vertical():
    """驗證垂直方向的三點共線"""
    p1 = Point(1, 0)
    p2 = Point(1, 5)
    p3 = Point(1, 10)
    assert not is_right_triangle(p1, p2, p3)

def test_invalid_input_type():
    """驗證輸入不是 Point 物件時是否拋出 TypeError"""
    p1 = Point(0, 0)
    p2 = Point(3, 0)
    with pytest.raises(TypeError):
        is_right_triangle(p1, p2, "not a point")