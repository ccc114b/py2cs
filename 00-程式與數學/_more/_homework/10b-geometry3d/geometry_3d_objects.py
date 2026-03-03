# geometry_3d_objects.py
import math

# 浮點數比較容忍度
EPSILON = 1e-9

class Point3D:
    """表示三維歐幾里得空間中的一個點 (x, y, z)"""
    def __init__(self, x, y, z):
        if not all(isinstance(coord, (int, float)) for coord in [x, y, z]):
            raise TypeError("Point3D coordinates must be numbers.")
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def __repr__(self):
        return f"Point3D({self.x}, {self.y}, {self.z})"

    def __eq__(self, other):
        if not isinstance(other, Point3D):
            return NotImplemented
        return math.isclose(self.x, other.x, abs_tol=EPSILON) and \
               math.isclose(self.y, other.y, abs_tol=EPSILON) and \
               math.isclose(self.z, other.z, abs_tol=EPSILON)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.x, self.y, self.z))

    # ------------------------------------------------------------------
    # 點的運算 (也可用於向量運算，將點視為從原點出發的向量)
    # ------------------------------------------------------------------

    def __add__(self, other): # 點 + 向量 = 點 (或 向量 + 向量 = 向量)
        if isinstance(other, Point3D):
            return Point3D(self.x + other.x, self.y + other.y, self.z + other.z)
        raise TypeError("Can only add Point3D to Point3D (as vector).")

    def __sub__(self, other): # 點 - 點 = 向量 (或 向量 - 向量 = 向量)
        if isinstance(other, Point3D):
            return Point3D(self.x - other.x, self.y - other.y, self.z - other.z)
        raise TypeError("Can only subtract Point3D from Point3D.")

    def __mul__(self, scalar): # 向量 * 純量 = 向量
        if not isinstance(scalar, (int, float)):
            raise TypeError("Can only multiply Point3D (as vector) by a scalar.")
        return Point3D(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar): # 純量 * 向量 = 向量
        return self.__mul__(scalar)

    def dot(self, other): # 內積 (點積)
        if not isinstance(other, Point3D):
            raise TypeError("Can only compute dot product with another Point3D (as vector).")
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other): # 叉積 (只適用於 3D 向量)
        if not isinstance(other, Point3D):
            raise TypeError("Can only compute cross product with another Point3D (as vector).")
        return Point3D(self.y * other.z - self.z * other.y,
                       self.z * other.x - self.x * other.z,
                       self.x * other.y - self.y * other.x)

    def magnitude(self): # 向量的長度 (範數)
        return math.sqrt(self.dot(self))

    def distance_to(self, other): # 點到點的距離
        if not isinstance(other, Point3D):
            raise TypeError("Can only compute distance to another Point3D.")
        diff_vector = self - other
        return diff_vector.magnitude()

    def normalize(self): # 單位向量
        mag = self.magnitude()
        if math.isclose(mag, 0, abs_tol=EPSILON):
            return Point3D(0, 0, 0) # 零向量沒有方向
        return Point3D(self.x / mag, self.y / mag, self.z / mag)

# 零點 / 零向量
ORIGIN_3D = Point3D(0, 0, 0)

# ------------------------------------------------------------------
# 線 (Line3D)
# ------------------------------------------------------------------
class Line3D:
    """表示三維空間中的一條直線，由兩個點定義"""
    def __init__(self, p1, p2):
        if not isinstance(p1, Point3D) or not isinstance(p2, Point3D):
            raise TypeError("Line3D must be defined by two Point3D objects.")
        if p1 == p2:
            raise ValueError("Two distinct points are required to define a line.")
        self.p1 = p1
        self.p2 = p2
        self.direction_vector = (p2 - p1).normalize() # 單位方向向量

    def __repr__(self):
        return f"Line3D({self.p1}, {self.p2})"

    def contains_point(self, p):
        """檢查點是否在直線上"""
        if not isinstance(p, Point3D):
            raise TypeError("Input must be a Point3D object.")
        # 點 p 在直線上，則向量 (p - p1) 與方向向量平行
        # 兩個向量平行，則它們的叉積為零向量
        vec_to_p = p - self.p1
        cross_product_mag = vec_to_p.cross(self.direction_vector).magnitude()
        return math.isclose(cross_product_mag, 0, abs_tol=EPSILON)

    def is_parallel_to(self, other_line):
        """檢查兩條直線是否平行"""
        if not isinstance(other_line, Line3D):
            raise TypeError("Input must be a Line3D object.")
        # 兩條直線平行，則它們的方向向量平行 (叉積為零向量)
        cross_product_mag = self.direction_vector.cross(other_line.direction_vector).magnitude()
        return math.isclose(cross_product_mag, 0, abs_tol=EPSILON)

    def is_perpendicular_to(self, other_line):
        """檢查兩條直線是否垂直"""
        if not isinstance(other_line, Line3D):
            raise TypeError("Input must be a Line3D object.")
        # 兩條直線垂直，則它們的方向向量內積為 0
        return math.isclose(self.direction_vector.dot(other_line.direction_vector), 0, abs_tol=EPSILON)

# ------------------------------------------------------------------
# 面 (Plane)
# ------------------------------------------------------------------
class Plane:
    """表示三維空間中的一個平面，由三個不共線的點定義"""
    def __init__(self, p1, p2, p3):
        if not all(isinstance(p, Point3D) for p in [p1, p2, p3]):
            raise TypeError("Plane must be defined by three Point3D objects.")
        
        v1 = p2 - p1
        v2 = p3 - p1
        
        # 法向量是 v1 和 v2 的叉積
        normal_unnormalized = v1.cross(v2)
        
        if math.isclose(normal_unnormalized.magnitude(), 0, abs_tol=EPSILON):
            raise ValueError("Three non-collinear points are required to define a plane.")
        
        self.normal_vector = normal_unnormalized.normalize() # 單位法向量
        self.p1 = p1 # 平面上的一個點

        # 平面方程 Ax + By + Cz + D = 0
        self.A = self.normal_vector.x
        self.B = self.normal_vector.y
        self.C = self.normal_vector.z
        self.D = - (self.A * p1.x + self.B * p1.y + self.C * p1.z)

    def __repr__(self):
        return f"Plane(point={self.p1}, normal={self.normal_vector})"

    def __eq__(self, other):
        if not isinstance(other, Plane):
            return NotImplemented
        # 兩個平面相等，如果它們的法向量平行且它們都包含同一個點
        # 或者更嚴謹，法向量平行且它們的平面方程等價
        # 檢查法向量是否平行 (方向可能相反)
        normals_parallel = math.isclose(self.normal_vector.cross(other.normal_vector).magnitude(), 0, abs_tol=EPSILON)
        
        # 檢查一個點是否在另一個平面上
        point_on_other_plane = other.contains_point(self.p1)
        
        return normals_parallel and point_on_other_plane

    def contains_point(self, p):
        """檢查點是否在平面上"""
        if not isinstance(p, Point3D):
            raise TypeError("Input must be a Point3D object.")
        # 點在平面上，則 (p - p1) 與法向量垂直 (內積為 0)
        vec_to_p = p - self.p1
        return math.isclose(vec_to_p.dot(self.normal_vector), 0, abs_tol=EPSILON)
        # 或者使用平面方程 Ax + By + Cz + D = 0
        # return math.isclose(self.A * p.x + self.B * p.y + self.C * p.z + self.D, 0, abs_tol=EPSILON)

    def is_parallel_to(self, other_plane):
        """檢查兩個平面是否平行"""
        if not isinstance(other_plane, Plane):
            raise TypeError("Input must be a Plane object.")
        # 平行則法向量平行 (叉積為零向量)
        cross_product_mag = self.normal_vector.cross(other_plane.normal_vector).magnitude()
        return math.isclose(cross_product_mag, 0, abs_tol=EPSILON)

    def is_perpendicular_to(self, other_plane):
        """檢查兩個平面是否垂直"""
        if not isinstance(other_plane, Plane):
            raise TypeError("Input must be a Plane object.")
        # 垂直則法向量垂直 (內積為 0)
        return math.isclose(self.normal_vector.dot(other_plane.normal_vector), 0, abs_tol=EPSILON)

# 輔助函式：檢查四個點是否共面
def are_points_coplanar(p1, p2, p3, p4):
    if not all(isinstance(p, Point3D) for p in [p1, p2, p3, p4]):
        raise TypeError("All inputs must be Point3D objects.")
    
    # 如果前三點共線，則無法定義唯一的平面
    v1 = p2 - p1
    v2 = p3 - p1
    if math.isclose(v1.cross(v2).magnitude(), 0, abs_tol=EPSILON):
        # 檢查所有四點是否共線
        if Line3D(p1, p2).contains_point(p3) and Line3D(p1, p2).contains_point(p4):
            return True # 四點共線，自然共面
        return False # 前三點共線，但第四點不共線，則無法共面
    
    # 前三點不共線，可以定義一個平面
    plane = Plane(p1, p2, p3)
    return plane.contains_point(p4)
