import random
from fractions import Fraction
from group import Group

class RationalAddGroup(Group):
    """有理數加法群 (Q, +)"""
    def __init__(self):
        self._identity = Fraction(0)
    
    @property
    def identity(self):
        return self._identity

    def operation(self, a, b):
        """有理數加法運算"""
        return Fraction(a) + Fraction(b)

    def inverse(self, val):
        """有理數加法逆元（負數）"""
        return -Fraction(val)

    def include(self, element):
        """檢查元素是否為有理數"""
        try:
            Fraction(element)
            return True
        except (TypeError, ValueError):
            return False

    def random_generate(self):
        """隨機生成一個有理數"""
        # 生成分子和分母，避免分母為0
        numerator = random.randint(-10, 10)
        denominator = random.randint(1, 10)
        return Fraction(numerator, denominator)

class RationalMulGroup(Group):
    """有理數乘法群 (Q*, ×) - 排除零元素"""
    def __init__(self):
        self._identity = Fraction(1)

    @property
    def identity(self):
        return self._identity

    def operation(self, a, b):
        """有理數乘法運算"""
        #if not (self.include(a) and self.include(b)):
        #    raise ValueError("Input elements must be non-zero rational numbers.")
        return Fraction(a) * Fraction(b)

    def inverse(self, val):
        """有理數乘法逆元（倒數）"""
        if not self.include(val):
            raise ValueError("Input element must be a non-zero rational number.")
        
        frac_val = Fraction(val)
        if frac_val == 0:
            raise ValueError("Zero has no multiplicative inverse.")
        
        return Fraction(1) / frac_val

    def include(self, element):
        """檢查元素是否為非零有理數"""
        try:
            frac_element = Fraction(element)
            return frac_element != 0
        except (TypeError, ValueError):
            return False

    def random_generate(self):
        """隨機生成一個非零有理數"""
        # 生成非零分子和分母
        numerator = random.choice([i for i in range(-10, 11) if i != 0])
        denominator = random.randint(1, 10)
        return Fraction(numerator, denominator)

# --- 有理數體物件 (組合模式) ---

class RationalField:
    """
    一個表示有理數體 Q 的類別，由加法群和乘法群組合而成。
    """
    def __init__(self):
        """
        初始化一個有理數體物件，並建立其加法群和乘法群物件。
        """
        # 組合模式：將加法群和乘法群物件作為屬性
        self.add_group = RationalAddGroup()
        self.mul_group = RationalMulGroup()

    # 透過內部物件呼叫對應的運算方法
    def add(self, a, b):
        """有理數加法"""
        return self.add_group.operation(a, b)

    def subtract(self, a, b):
        """有理數減法 (a - b = a + (-b))"""
        return self.add_group.operation(a, self.add_group.inverse(b))
        
    def multiply(self, a, b):
        """有理數乘法"""
        return self.mul_group.operation(a, b)
    
    def divide(self, a, b):
        """有理數除法 (a ÷ b = a × b^(-1))"""
        return self.mul_group.operation(a, self.mul_group.inverse(b))

# 範例使用
if __name__ == "__main__":
    # 建立一個有理數體的物件
    Q = RationalField()
    print("--- 測試有理數體 Q ---")
    
    # 加法測試
    a = Q.random_rational()
    b = Q.random_rational()
    add_result = Q.add(a, b)
    print(f"加法: {a} + {b} = {add_result}")
    
    # 減法測試
    sub_result = Q.subtract(a, b)
    print(f"減法: {a} - {b} = {sub_result}")
    
    # 乘法測試
    c = Q.random_nonzero_rational()
    d = Q.random_nonzero_rational()
    mul_result = Q.multiply(c, d)
    print(f"乘法: {c} × {d} = {mul_result}")
    
    # 除法測試
    div_result = Q.divide(c, d)
    print(f"除法: {c} ÷ {d} = {div_result}")

    # 分配律測試 (此測試需要兩個運算)
    e = Q.random_nonzero_rational()
    f = Q.random_rational()
    g = Q.random_rational()
    
    # 檢驗 e × (f + g) = (e × f) + (e × g)
    lhs = Q.multiply(e, Q.add(f, g))
    rhs = Q.add(Q.multiply(e, f), Q.multiply(e, g))
    print(f"\n分配律測試: {e} × ({f} + {g}) = {lhs}")
    print(f"({e} × {f}) + ({e} × {g}) = {rhs}")
    assert lhs == rhs, "分配律測試失敗！"
    print("分配律測試成功！")
    
    # 逆元測試
    print(f"\n逆元測試:")
    test_val = Q.random_nonzero_rational()
    add_inv = Q.additive_inverse(test_val)
    mul_inv = Q.multiplicative_inverse(test_val)
    
    print(f"{test_val} 的加法逆元: {add_inv}")
    print(f"驗證: {test_val} + ({add_inv}) = {Q.add(test_val, add_inv)}")
    
    print(f"{test_val} 的乘法逆元: {mul_inv}")
    print(f"驗證: {test_val} × {mul_inv} = {Q.multiply(test_val, mul_inv)}")