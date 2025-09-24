# field_elements.py
from fractions import Fraction
import math

# 我們的集合 F 是有理數 Q
# 加法運算
def field_add(a, b):
    return a + b

# 乘法運算
def field_multiply(a, b):
    return a * b

# 加法單位元素 (0)
ADDITIVE_IDENTITY = Fraction(0, 1)

# 乘法單位元素 (1)
MULTIPLICATIVE_IDENTITY = Fraction(1, 1)

# 判斷元素是否在體 F 中 (這裡簡化為判斷是否為 Fraction 型別)
def is_in_field_F(element):
    return isinstance(element, Fraction)

# 取得加法反元素
def get_additive_inverse(a):
    return -a

# 取得乘法反元素 (a 必須不為 0)
def get_multiplicative_inverse(a):
    if a == ADDITIVE_IDENTITY:
        raise ValueError("Cannot get multiplicative inverse of zero.")
    return Fraction(1, a)

