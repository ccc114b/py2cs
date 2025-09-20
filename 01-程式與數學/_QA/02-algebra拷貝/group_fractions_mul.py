import random # 用來產生隨機測試數據
import fractions

op = '*'
identity = fractions.Fraction(1, 1) # 單位元素 (乘法的單位元素是 1)

# 我們的集合是整數，Python 的 int 型別天然地表示整數。
# 我們的運算是乘法。
def operation(a, b):
    """模擬群的二元運算 (這裡使用乘法)"""
    return a * b

def inverse(val):
    return identity/val # 乘法的反元素

TEST_RANGE = 100
# 輔助函式，生成隨機整數
def random_generate():
    a = random.randint(-TEST_RANGE, TEST_RANGE)
    b = random.randint(-TEST_RANGE, TEST_RANGE)
    if a == 0:
        a = 1 # 避免分子為零(因為沒有反元素，0 在乘法群中不被包含)
    if b == 0:
        b = 1 # 避免分母為零（會產生例外錯誤）
    return fractions.Fraction(a, b)

# 我們可以假設有一個函式來檢查一個數是否屬於我們的集合 G (整數)
# 對於 Python 的 int 型別，這幾乎總是 True，除非是 overflow
def is_in_group(element):
    """檢查元素是否屬於我們的集合 G (這裡簡化為檢查是否為整數)"""
    return isinstance(element, fractions.Fraction)