import random
from group import Group

class FiniteFieldAddGroup(Group):
    def __init__(self, modulus):
        self.MODULUS = modulus
        self._identity = 0
    
    @property
    def identity(self):
        return self._identity

    def operation(self, a, b):
        return (a + b) % self.MODULUS

    def inverse(self, val):
        return (-val) % self.MODULUS

    def include(self, element):
        return isinstance(element, int) and 0 <= element < self.MODULUS

    def random_generate(self):
        return random.randint(0, self.MODULUS - 1)

class FiniteFieldMulGroup(Group):
    def __init__(self, modulus):
        self.MODULUS = modulus
        self._identity = 1

    @property
    def identity(self):
        return self._identity

    def _extended_gcd(self, a, b):
        if a == 0:
            return (b, 0, 1)
        d, x1, y1 = self._extended_gcd(b % a, a)
        x = y1 - (b // a) * x1
        y = x1
        return (d, x, y)

    def operation(self, a, b):
        #if not (self.include(a) and self.include(b)):
        #    raise ValueError("Input elements must be in the group.")
        return (a * b) % self.MODULUS

    def inverse(self, val):
        if not self.include(val):
            raise ValueError("Input element must be in the group.")
        if val == 0:
            raise ValueError("0 has no multiplicative inverse.")
        
        g, x, y = self._extended_gcd(val, self.MODULUS)
        
        if g != 1:
            raise ValueError(f"Inverse does not exist for {val} in GF({self.MODULUS}).")
            
        return x % self.MODULUS

    def include(self, element):
        return isinstance(element, int) and 1 <= element < self.MODULUS

    def random_generate(self):
        return random.randint(1, self.MODULUS - 1)

# --- 有限體物件 (組合模式) ---

class FiniteField:
    """
    一個表示有限體 GF(p) 的類別，由加法群和乘法群組合而成。
    """
    def __init__(self, modulus):
        """
        初始化一個有限體物件，並建立其加法群和乘法群物件。
        """
        if not isinstance(modulus, int) or modulus <= 1:
            raise ValueError("Modulus must be a prime number greater than 1.")
        
        self.MODULUS = modulus
        # 組合模式：將加法群和乘法群物件作為屬性
        self.add_group = FiniteFieldAddGroup(modulus)
        self.mul_group = FiniteFieldMulGroup(modulus)

    # 透過內部物件呼叫對應的運算方法
    def add(self, a, b):
        return self.add_group.operation(a, b)

    def substract(self, val):
        return self.add_group.inverse(val)
        
    def multiply(self, a, b):
        return self.mul_group.operation(a, b)
    
    def divide(self, val):
        return self.mul_group.inverse(val)

    # 輔助函式，用於檢驗分配律等
    def include(self, element):
        return self.add_group.include(element)
        
    def random_element(self):
        return self.add_group.random_generate()

# 範例使用
if __name__ == "__main__":
    # 建立一個 GF(7) 的物件
    gf7 = FiniteField(7)
    print("--- 測試 GF(7) ---")
    
    # 加法測試
    a = gf7.random_element()
    b = gf7.random_element()
    add_result = gf7.add(a, b)
    print(f"加法: {a} + {b} (mod 7) = {add_result}")
    
    # 乘法測試
    c = gf7.random_element()
    d = gf7.random_element()
    mul_result = gf7.multiply(c, d)
    print(f"乘法: {c} * {d} (mod 7) = {mul_result}")

    # 分配律測試 (此測試需要兩個運算)
    e = gf7.random_element()
    f = gf7.random_element()
    g = gf7.random_element()
    
    # 檢驗 e * (f + g) = (e * f) + (e * g)
    lhs = gf7.multiply(e, gf7.add(f, g))
    rhs = gf7.add(gf7.multiply(e, f), gf7.multiply(e, g))
    print(f"\n分配律測試: {e} * ({f} + {g}) = {lhs}")
    print(f"({e} * {f}) + ({e} * {g}) = {rhs}")
    assert lhs == rhs, "分配律測試失敗！"
    print("分配律測試成功！")