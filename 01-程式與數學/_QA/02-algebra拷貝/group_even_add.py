import random # 用來產生隨機測試數據

op = '+'
identity = 0 # 單位元素 (加法的單位元素是 0
# 我們的集合是整數，Python 的 int 型別天然地表示整數。
# 我們的運算是乘馬。
def operation(a, b):
    """模擬群的二元運算 (這裡使用加法)"""
    return a + b

def inverse(val):
    return -val # 加法的反元素

TEST_RANGE = 100
# 輔助函式，生成隨機整數
def random_generate():
    return random.randint(-TEST_RANGE, TEST_RANGE)*2

# 我們可以假設有一個函式來檢查一個數是否屬於我們的集合 G (整數)
# 對於 Python 的 int 型別，這幾乎總是 True，除非是 overflow
def is_in_group(element):
    """檢查元素是否屬於我們的集合 G (這裡簡化為檢查是否為整數)"""
    return isinstance(element, int) and element % 2 == 0