# geometry_objects.py
import math

# 浮點數比較容忍度
EPSILON = 1e-9

class Point:
    """表示二維歐幾里得空間中的一個點 (x, y)"""
    def __init__(self, x, y):
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            raise TypeError("Point coordinates must be numbers.")
        self.x = float(x)
        self.y = float(y)

    def __repr__(self):
        return f"Point({self.x}, {self.y})"

    def __eq__(self, other):
        if not isinstance(other, Point):
            return NotImplemented
        return math.isclose(self.x, other.x, abs_tol=EPSILON) and \
               math.isclose(self.y, other.y, abs_tol=EPSILON)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self): # 讓點可以作為字典的鍵或集合的元素
        return hash((self.x, self.y))

    # ------------------------------------------------------------------
    # 點的運算 (也可用於向量運算，將點視為從原點出發的向量)
    # ------------------------------------------------------------------

    def __add__(self, other): # 點 + 向量 = 點 (或 向量 + 向量 = 向量)
        if isinstance(other, Point): # 這裡我們將點視為向量來處理
            return Point(self.x + other.x, self.y + other.y)
        raise TypeError("Can only add Point to Point (as vector).")

    def __sub__(self, other): # 點 - 點 = 向量 (或 向量 - 向量 = 向量)
        if isinstance(other, Point):
            return Point(self.x - other.x, self.y - other.y)
        raise TypeError("Can only subtract Point from Point.")

    def __mul__(self, scalar): # 向量 * 純量 = 向量
        if not isinstance(scalar, (int, float)):
            raise TypeError("Can only multiply Point (as vector) by a scalar.")
        return Point(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar): # 純量 * 向量 = 向量
        return self.__mul__(scalar)

    def dot(self, other): # 內積 (點積)
        if not isinstance(other, Point): # 這裡假設 other 也是一個表示向量的 Point
            raise TypeError("Can only compute dot product with another Point (as vector).")
        return self.x * other.x + self.y * other.y

    def magnitude(self): # 向量的長度 (範數)
        return math.sqrt(self.dot(self))

    def distance_to(self, other): # 點到點的距離
        if not isinstance(other, Point):
            raise TypeError("Can only compute distance to another Point.")
        diff_vector = self - other
        return diff_vector.magnitude()

    def normalize(self): # 單位向量
        mag = self.magnitude()
        if mag == 0:
            return Point(0, 0) # 零向量沒有方向
        return Point(self.x / mag, self.y / mag)

# 零點 / 零向量
ORIGIN = Point(0, 0)

# ------------------------------------------------------------------
# 線 (Line)
# ------------------------------------------------------------------
class Line:
    """表示二維空間中的一條直線，由兩個點定義"""
    def __init__(self, p1, p2):
        if not isinstance(p1, Point) or not isinstance(p2, Point):
            raise TypeError("Line must be defined by two Point objects.")
        if p1 == p2:
            raise ValueError("Two distinct points are required to define a line.")
        self.p1 = p1
        self.p2 = p2
        self.direction_vector = (p2 - p1).normalize() # 方向向量
        # 直線的標準式 Ax + By + C = 0
        # A = y2 - y1
        # B = x1 - x2
        # C = -A*x1 - B*y1
        self.A = p2.y - p1.y
        self.B = p1.x - p2.x
        self.C = -self.A * p1.x - self.B * p1.y

    def __repr__(self):
        return f"Line({self.p1}, {self.p2})"

    def contains_point(self, p):
        """檢查點是否在直線上"""
        if not isinstance(p, Point):
            raise TypeError("Input must be a Point object.")
        # 點在直線上，則 (p - p1) 與 direction_vector 平行
        # 也就是 (p - p1) 的方向向量等於 direction_vector 或 -direction_vector
        # 或者使用標準式 Ax + By + C = 0
        return math.isclose(self.A * p.x + self.B * p.y + self.C, 0, abs_tol=EPSILON)

    def is_parallel_to(self, other_line):
        """檢查兩條直線是否平行"""
        if not isinstance(other_line, Line):
            raise TypeError("Input must be a Line object.")
        # 兩條直線平行，則它們的方向向量平行 (內積的絕對值等於兩向量長度乘積)
        # 或者更簡單，方向向量的叉積為0 (在2D中，叉積是標量)
        # 2D 向量 (x1, y1) 和 (x2, y2) 的「叉積」是 x1*y2 - x2*y1
        # 或者檢查 A/B 比值是否相等
        return math.isclose(self.direction_vector.x * other_line.direction_vector.y - \
                            self.direction_vector.y * other_line.direction_vector.x, 0, abs_tol=EPSILON)

    def is_perpendicular_to(self, other_line):
        """檢查兩條直線是否垂直"""
        if not isinstance(other_line, Line):
            raise TypeError("Input must be a Line object.")
        # 兩條直線垂直，則它們的方向向量內積為 0
        return math.isclose(self.direction_vector.dot(other_line.direction_vector), 0, abs_tol=EPSILON)

# ------------------------------------------------------------------
# 面 (Plane) - 在二維空間中，面就是整個平面，通常不單獨定義一個 Plane 類別。
# 但如果我們擴展到三維，就需要 Plane 類別。
# 這裡我們暫時不定義 Plane 類別，因為我們專注於 2D。
# ------------------------------------------------------------------