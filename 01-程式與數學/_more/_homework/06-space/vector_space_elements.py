# vector_space_elements.py
import math

class Vector2D:
    """表示 R^2 空間中的一個向量"""
    def __init__(self, x, y):
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            raise TypeError("Vector components must be numbers.")
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Vector2D({self.x}, {self.y})"

    def __eq__(self, other):
        if not isinstance(other, Vector2D):
            return NotImplemented
        # 浮點數比較需要容忍度
        return math.isclose(self.x, other.x) and math.isclose(self.y, other.y)

    def __add__(self, other): # 向量加法
        if not isinstance(other, Vector2D):
            raise TypeError("Can only add Vector2D to Vector2D.")
        return Vector2D(self.x + other.x, self.y + other.y)

    def __sub__(self, other): # 向量減法 (用於距離計算)
        if not isinstance(other, Vector2D):
            raise TypeError("Can only subtract Vector2D from Vector2D.")
        return Vector2D(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar): # 純量乘法 (右乘)
        if not isinstance(scalar, (int, float)):
            raise TypeError("Can only multiply Vector2D by a scalar.")
        return Vector2D(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar): # 純量乘法 (左乘)
        return self.__mul__(scalar)

    # 歐幾里得範數 (L2 範數)
    def norm(self):
        return math.sqrt(self.x**2 + self.y**2)

# 零向量 (加法單位元素)
ZERO_VECTOR = Vector2D(0, 0)

# 檢查是否為向量 (簡化為檢查型別)
def is_vector_in_space(obj):
    return isinstance(obj, Vector2D)

# 檢查是否為純量 (實數)
def is_scalar(obj):
    return isinstance(obj, (int, float))
