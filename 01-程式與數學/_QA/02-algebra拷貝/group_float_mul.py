import random # 用來產生隨機測試數據

op = '*'
identity = 1.0 # 單位元素 (乘法的單位元素是 1.0)

# 我們的集合是整數，Python 的 int 型別天然地表示整數。
# 我們的運算是乘法。
def operation(a, b):
    """模擬群的二元運算 (這裡使用乘法)"""
    return a * b

def inverse(val):
    return 1.0/val # 乘馬的反元素

TEST_RANGE = 100
# 輔助函式，生成隨機整數
def random_generate():
    r = random.uniform(-TEST_RANGE, TEST_RANGE)
    if r == 0.0:
        r = 1.0 # 避免產生 0.0 (因為沒有反元素，0 在乘法群中不被包含)
    return r

# 我們可以假設有一個函式來檢查一個數是否屬於我們的集合 G (整數)
# 對於 Python 的 int 型別，這幾乎總是 True，除非是 overflow
def is_in_group(element):
    """檢查元素是否屬於我們的集合 G (這裡簡化為檢查是否為浮點數)"""
    return isinstance(element, float)
