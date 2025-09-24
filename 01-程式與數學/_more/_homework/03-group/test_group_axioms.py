# test_group_axioms.py
import pytest
import random # 用來產生隨機測試數據
from group_elements import group_operation, IDENTITY_ELEMENT, is_in_set_G

# 設定測試用的隨機整數範圍
TEST_RANGE = 100
NUM_TEST_CASES = 100

# 輔助函式，生成隨機整數
def get_random_int():
    return random.randint(-TEST_RANGE, TEST_RANGE)

# 1. 封閉性 (Closure)
def test_closure():
    for _ in range(NUM_TEST_CASES):
        a = get_random_int()
        b = get_random_int()
        result = group_operation(a, b)
        assert is_in_set_G(result), f"Closure failed: {a} + {b} = {result} is not in G"

# 2. 結合性 (Associativity)
def test_associativity():
    for _ in range(NUM_TEST_CASES):
        a = get_random_int()
        b = get_random_int()
        c = get_random_int()
        assert group_operation(group_operation(a, b), c) == group_operation(a, group_operation(b, c)), \
            f"Associativity failed: ({a} + {b}) + {c} != {a} + ({b} + {c})"

# 3. 單位元素 (Identity Element)
def test_identity_element():
    for _ in range(NUM_TEST_CASES):
        a = get_random_int()
        # 左單位元素
        assert group_operation(a, IDENTITY_ELEMENT) == a, \
            f"Left identity failed: {a} + {IDENTITY_ELEMENT} != {a}"
        # 右單位元素
        assert group_operation(IDENTITY_ELEMENT, a) == a, \
            f"Right identity failed: {IDENTITY_ELEMENT} + {a} != {a}"

# 4. 反元素 (Inverse Element)
def test_inverse_element():
    for _ in range(NUM_TEST_CASES):
        a = get_random_int()
        # 加法的反元素是負號
        # 這裡我們需要一個函式來計算反元素，因為它不是固定的運算
        def get_inverse(val):
            return -val # 加法的反元素

        a_inverse = get_inverse(a)
        
        # 檢查反元素是否也在集合 G 中 (對於整數，-a 仍然是整數)
        assert is_in_set_G(a_inverse), f"Inverse {a_inverse} for {a} is not in G"

        # 檢查左反元素
        assert group_operation(a, a_inverse) == IDENTITY_ELEMENT, \
            f"Left inverse failed: {a} + {a_inverse} != {IDENTITY_ELEMENT}"
        # 檢查右反元素
        assert group_operation(a_inverse, a) == IDENTITY_ELEMENT, \
            f"Right inverse failed: {a_inverse} + {a} != {IDENTITY_ELEMENT}"
