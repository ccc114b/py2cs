# group_elements.py

# 我們的集合是整數，Python 的 int 型別天然地表示整數。
# 我們的運算是加法。
def group_operation(a, b):
    """模擬群的二元運算 (這裡使用加法)"""
    return a + b

# 單位元素 (加法的單位元素是 0)
IDENTITY_ELEMENT = 0

# 我們可以假設有一個函式來檢查一個數是否屬於我們的集合 G (整數)
# 對於 Python 的 int 型別，這幾乎總是 True，除非是 overflow
def is_in_set_G(element):
    """檢查元素是否屬於我們的集合 G (這裡簡化為檢查是否為整數)"""
    return isinstance(element, int)