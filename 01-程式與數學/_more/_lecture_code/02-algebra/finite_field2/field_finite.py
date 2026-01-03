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
        return (a * b) % self.MODULUS

    def inverse(self, val):
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
        if not isinstance(modulus, int) or modulus <= 1:
            raise ValueError("Modulus must be a prime number greater than 1.")
        
        self.MODULUS = modulus
        self.add_group = FiniteFieldAddGroup(modulus)
        self.mul_group = FiniteFieldMulGroup(modulus)

    def add(self, a, b):
        return self.add_group.operation(a, b)

    def substract(self, val):
        return self.add_group.inverse(val)
        
    def multiply(self, a, b):
        return self.mul_group.operation(a, b)
    
    def divide(self, val):
        return self.mul_group.inverse(val)

    def include(self, element):
        return self.add_group.include(element)
        
    def random_element(self):
        return self.add_group.random_generate()

    def element(self, value):
        """
        一個工廠方法，用於創建 FiniteFieldElement 物件
        並將其與當前的有限體實例綁定。
        """
        return FiniteFieldElement(value, self)

class FiniteFieldElement:
    """
    表示有限體中的一個元素，並支援運算子重載。
    """
    def __init__(self, value, field):
        if not field.include(value):
            raise ValueError(f"Value {value} is not in the field GF({field.MODULUS}).")
        
        self.value = value
        self.field = field

    def __repr__(self):
        return f"GF({self.field.MODULUS}).element({self.value})"

    def __eq__(self, other):
        """
        相等性比較
        """
        if isinstance(other, FiniteFieldElement):
            return self.value == other.value and self.field.MODULUS == other.field.MODULUS
        return False
        
    def __add__(self, other):
        """
        加法運算 (+)
        """
        if isinstance(other, FiniteFieldElement):
            if self.field.MODULUS != other.field.MODULUS:
                raise ValueError("Cannot add elements from different fields.")
            new_value = self.field.add(self.value, other.value)
            return self.field.element(new_value)
        
        # 支援與普通整數的運算
        if isinstance(other, int):
            new_value = self.field.add(self.value, other)
            return self.field.element(new_value)
        
        raise TypeError(f"Unsupported operand type for +: 'FiniteFieldElement' and '{type(other).__name__}'")

    def __sub__(self, other):
        """
        減法運算 (-)
        """
        if isinstance(other, FiniteFieldElement):
            if self.field.MODULUS != other.field.MODULUS:
                raise ValueError("Cannot subtract elements from different fields.")
            other_inverse = self.field.substract(other.value)
            new_value = self.field.add(self.value, other_inverse)
            return self.field.element(new_value)

        if isinstance(other, int):
            other_inverse = self.field.substract(other)
            new_value = self.field.add(self.value, other_inverse)
            return self.field.element(new_value)
            
        raise TypeError(f"Unsupported operand type for -: 'FiniteFieldElement' and '{type(other).__name__}'")

    def __mul__(self, other):
        """
        乘法運算 (*)
        """
        if isinstance(other, FiniteFieldElement):
            if self.field.MODULUS != other.field.MODULUS:
                raise ValueError("Cannot multiply elements from different fields.")
            new_value = self.field.multiply(self.value, other.value)
            return self.field.element(new_value)
            
        if isinstance(other, int):
            new_value = self.field.multiply(self.value, other)
            return self.field.element(new_value)
            
        raise TypeError(f"Unsupported operand type for *: 'FiniteFieldElement' and '{type(other).__name__}'")
        
    def __truediv__(self, other):
        """
        除法運算 (/)
        """
        if isinstance(other, FiniteFieldElement):
            if self.field.MODULUS != other.field.MODULUS:
                raise ValueError("Cannot divide elements from different fields.")
            other_inverse = self.field.divide(other.value)
            new_value = self.field.multiply(self.value, other_inverse)
            return self.field.element(new_value)

        if isinstance(other, int):
            other_inverse = self.field.divide(other)
            new_value = self.field.multiply(self.value, other_inverse)
            return self.field.element(new_value)

        raise TypeError(f"Unsupported operand type for /: 'FiniteFieldElement' and '{type(other).__name__}'")
        
    def __neg__(self):
        """
        一元負號運算 (-)
        """
        new_value = self.field.substract(self.value)
        return self.field.element(new_value)

# 範例使用
if __name__ == "__main__":
    # 建立一個 GF(7) 的物件
    gf7 = FiniteField(7)
    
    # 使用新的 element 工廠方法來建立元素
    a = gf7.element(3)
    b = gf7.element(5)
    
    print("--- 測試 FiniteFieldElement 類別 ---")
    print(f"a = {a}")
    print(f"b = {b}")

    # 加法測試
    add_result = a + b
    print(f"加法: {a} + {b} = {add_result}")
    
    # 減法測試
    sub_result = a - b
    print(f"減法: {a} - {b} = {sub_result}")

    # 乘法測試
    mul_result = a * b
    print(f"乘法: {a} * {b} = {mul_result}")
    
    # 除法測試
    div_result = a / b
    print(f"除法: {a} / {b} = {div_result}")
    
    # 分配律測試 (使用新的運算子)
    c = gf7.element(2)
    d = gf7.element(3)
    e = gf7.element(4)
    
    # 檢驗 c * (d + e) = (c * d) + (c * e)
    lhs = c * (d + e)
    rhs = (c * d) + (c * e)
    print(f"\n分配律測試: {c} * ({d} + {e}) = {lhs}")
    print(f"({c} * {d}) + ({c} * {e}) = {rhs}")
    assert lhs == rhs, "分配律測試失敗！"
    print("分配律測試成功！")
    
    # 測試與整數的運算
    print("\n--- 測試與整數的混合運算 ---")
    f = gf7.element(4)
    g = f + 5
    h = f * 2
    print(f"4 + 5 (mod 7) = {g}")
    print(f"4 * 2 (mod 7) = {h}")
