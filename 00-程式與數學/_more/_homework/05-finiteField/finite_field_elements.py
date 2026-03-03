# finite_field_elements.py
import math

class FiniteFieldElement:
    """
    表示一個有限體 GF(p) 中的元素
    p 必須是一個質數
    """
    def __init__(self, value, modulus):
        if not isinstance(modulus, int) or modulus <= 1:
            raise ValueError("Modulus must be an integer greater than 1.")
        # 簡單檢查模數是否為質數 (對於小型質數足夠)
        if modulus == 2 or (modulus > 2 and modulus % 2 != 0 and
                            all(modulus % i != 0 for i in range(3, int(modulus**0.5) + 1, 2))):
            pass # 是質數，或通過簡單檢查
        else:
            # 這裡可以加入更嚴格的質數檢定，但為了範例簡化
            # print(f"Warning: Modulus {modulus} might not be a prime number. For GF(p) p must be prime.")
            pass # 允許非質數，但測試時我們確保會用質數

        self.value = value % modulus
        self.modulus = modulus

    def __repr__(self):
        return f"GF({self.modulus})({self.value})"

    def __eq__(self, other):
        if not isinstance(other, FiniteFieldElement):
            return NotImplemented
        return self.value == other.value and self.modulus == other.modulus

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.value, self.modulus))

    # --- 加法相關運算 ---
    def __add__(self, other):
        if not isinstance(other, FiniteFieldElement) or self.modulus != other.modulus:
            raise TypeError("Operands must be FiniteFieldElement of the same modulus.")
        return FiniteFieldElement(self.value + other.value, self.modulus)

    def __neg__(self): # 加法反元素
        return FiniteFieldElement(-self.value, self.modulus)

    # --- 乘法相關運算 ---
    def __mul__(self, other):
        if not isinstance(other, FiniteFieldElement) or self.modulus != other.modulus:
            raise TypeError("Operands must be FiniteFieldElement of the same modulus.")
        return FiniteFieldElement(self.value * other.value, self.modulus)

    def __truediv__(self, other): # 除法
        return self * other.multiplicative_inverse()

    def multiplicative_inverse(self): # 乘法反元素
        if self.value == 0:
            raise ValueError("Zero has no multiplicative inverse in a field.")
        # 使用擴展歐幾里得演算法 (Extended Euclidean Algorithm) 尋找模反元素
        # ax + by = gcd(a, b)
        # 對於模反元素，我們尋找 ax + pm = 1 (mod p)，其中 x 就是 a 的反元素
        a, m = self.value, self.modulus
        m0, x0, x1 = m, 0, 1
        while a > 1:
            q = a // m0
            m0, a = a % m0, m0
            x0, x1 = x1 - q * x0, x0
        return FiniteFieldElement(x1 % m, self.modulus)

# 定義針對 GF(p) 的特殊元素和判斷函式
def get_additive_identity(modulus):
    return FiniteFieldElement(0, modulus)

def get_multiplicative_identity(modulus):
    return FiniteFieldElement(1, modulus)

def is_in_field_F(element, modulus):
    return isinstance(element, FiniteFieldElement) and element.modulus == modulus