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
            return False # NotImplemented
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
# 圓 (Circle)
# ------------------------------------------------------------------
class Circle:
    """表示二維空間中的一個圓，由圓心和半徑定義"""
    def __init__(self, center, radius):
        if not isinstance(center, Point):
            raise TypeError("Circle center must be a Point object.")
        if not isinstance(radius, (int, float)):
            raise TypeError("Circle radius must be a number.")
        if radius <= 0:
            raise ValueError("Circle radius must be positive.")

        self.center = center
        self.radius = float(radius)
        self.radius_sq = self.radius ** 2 # 預先計算半徑的平方，避免重複計算，提高效率

    def __repr__(self):
        return f"Circle({self.center}, radius={self.radius})"

    def __eq__(self, other):
        if not isinstance(other, Circle):
            return False # NotImplemented
        # 兩個圓相等，若且唯若它們的圓心相同且半徑相同
        return self.center == other.center and \
               math.isclose(self.radius, other.radius, abs_tol=EPSILON)

    def __ne__(self, other):
        return not self.__eq__(other)

    def area(self):
        """計算圓的面積: π * r^2"""
        return math.pi * self.radius_sq

    def circumference(self):
        """計算圓的周長: 2 * π * r"""
        return 2 * math.pi * self.radius

    def contains_point(self, p):
        """檢查一個點是否在圓內（包括邊界上）"""
        if not isinstance(p, Point):
            raise TypeError("Input must be a Point object.")
        # 點 p 到圓心 center 的距離如果小於或等於半徑，則點在圓內或圓周上
        distance_squared = self.center.distance_to(p)**2 # 使用平方距離避免開根號，提高效率
        return distance_squared <= (self.radius_sq + EPSILON) # 考慮浮點數誤差

    def intersects_line(self, line):
        """檢查圓是否與一條直線相交"""
        if not isinstance(line, Line):
            raise TypeError("Input must be a Line object.")

        # 直線的一般式：Ax + By + C = 0
        # 圓心 (cx, cy) 到直線的距離公式：d = |A*cx + B*cy + C| / sqrt(A^2 + B^2)
        numerator = abs(line.A * self.center.x + line.B * self.center.y + line.C)
        denominator = math.sqrt(line.A**2 + line.B**2)

        # 避免除以零，儘管理論上 Line 構造函數已經保證 A 和 B 不會同時為零
        if math.isclose(denominator, 0, abs_tol=EPSILON):
            # 這表示 Line 的 A 和 B 都是 0，但在 Line 的構造函數中已經處理了這種情況
            # 實際上這不會發生，除非 Line 的定義有問題
            return False

        distance_to_line = numerator / denominator

        # 如果圓心到直線的距離 d <= 半徑 r，則圓與直線相交
        return distance_to_line <= (self.radius + EPSILON) # 考慮浮點數誤差

    def intersects_circle(self, other_circle):
        """檢查圓是否與另一個圓相交"""
        if not isinstance(other_circle, Circle):
            raise TypeError("Input must be a Circle object.")

        # 兩圓心的距離
        distance_between_centers = self.center.distance_to(other_circle.center)

        # 兩圓相交有三種情況：
        # 1. 兩圓相離：距離 > 半徑和 (沒有交點)
        # 2. 兩圓相切：距離 == 半徑和 (一個交點)
        # 3. 兩圓相交：距離 < 半徑和 (兩個交點)
        # 4. 一圓在另一圓內部，但不相交：距離 < 半徑差的絕對值 (沒有交點)
        # 5. 一圓在另一圓內部，且內切：距離 == 半徑差的絕對值 (一個交點)
        #
        # 總結來說，只要兩圓心的距離小於或等於它們半徑之和，就視為相交 (包括外切和內切)。
        # 但還要排除一個圓完全在另一個圓內部且不接觸的情況。
        # 也就是說，距離要滿足 abs(r1 - r2) <= distance <= r1 + r2
        # 最簡單判斷相交是： distance <= r1 + r2

        sum_of_radii = self.radius + other_circle.radius
        diff_of_radii = abs(self.radius - other_circle.radius)

        # 考慮浮點數誤差
        return (distance_between_centers <= (sum_of_radii + EPSILON)) and \
               (distance_between_centers >= (diff_of_radii - EPSILON))

