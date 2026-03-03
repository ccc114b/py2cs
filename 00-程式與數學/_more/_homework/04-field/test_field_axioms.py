# test_field_axioms.py
import pytest
import random
from fractions import Fraction
from field_elements import (
    field_add, field_multiply,
    ADDITIVE_IDENTITY, MULTIPLICATIVE_IDENTITY,
    is_in_field_F,
    get_additive_inverse, get_multiplicative_inverse
)

# 測試用的隨機有理數範圍
# 生成分子和分母在 -100 到 100 之間 (分母不為0)
TEST_RANGE = 100
NUM_TEST_CASES = 50

def get_random_fraction():
    numerator = random.randint(-TEST_RANGE, TEST_RANGE)
    denominator = random.randint(1, TEST_RANGE) # 分母不為0
    # 隨機決定正負
    if random.random() < 0.5:
        denominator = -denominator
    return Fraction(numerator, denominator)

def get_random_nonzero_fraction():
    while True:
        f = get_random_fraction()
        if f != ADDITIVE_IDENTITY:
            return f

# --- 第一組：關於加法，形成一個交換群 (F, +) ---

# 1. 加法封閉性
def test_additive_closure():
    for _ in range(NUM_TEST_CASES):
        a = get_random_fraction()
        b = get_random_fraction()
        result = field_add(a, b)
        assert is_in_field_F(result), f"Additive closure failed: {a} + {b} = {result} is not in F"

# 2. 加法結合性
def test_additive_associativity():
    for _ in range(NUM_TEST_CASES):
        a = get_random_fraction()
        b = get_random_fraction()
        c = get_random_fraction()
        assert field_add(field_add(a, b), c) == field_add(a, field_add(b, c)), \
            f"Additive associativity failed: ({a} + {b}) + {c} != {a} + ({b} + {c})"

# 3. 加法單位元素
def test_additive_identity():
    for _ in range(NUM_TEST_CASES):
        a = get_random_fraction()
        assert field_add(a, ADDITIVE_IDENTITY) == a, \
            f"Additive identity failed (right): {a} + {ADDITIVE_IDENTITY} != {a}"
        assert field_add(ADDITIVE_IDENTITY, a) == a, \
            f"Additive identity failed (left): {ADDITIVE_IDENTITY} + {a} != {a}"

# 4. 加法反元素
def test_additive_inverse():
    for _ in range(NUM_TEST_CASES):
        a = get_random_fraction()
        a_inverse = get_additive_inverse(a)
        assert is_in_field_F(a_inverse), f"Additive inverse {a_inverse} for {a} is not in F"
        assert field_add(a, a_inverse) == ADDITIVE_IDENTITY, \
            f"Additive inverse failed (right): {a} + {a_inverse} != {ADDITIVE_IDENTITY}"
        assert field_add(a_inverse, a) == ADDITIVE_IDENTITY, \
            f"Additive inverse failed (left): {a_inverse} + {a} != {ADDITIVE_IDENTITY}"

# 5. 加法交換性
def test_additive_commutativity():
    for _ in range(NUM_TEST_CASES):
        a = get_random_fraction()
        b = get_random_fraction()
        assert field_add(a, b) == field_add(b, a), \
            f"Additive commutativity failed: {a} + {b} != {b} + {a}"

# --- 第二組：關於乘法，形成一個非零元素的交換群 (F \ {0}, x) ---

# 6. 乘法封閉性
def test_multiplicative_closure():
    for _ in range(NUM_TEST_CASES):
        a = get_random_fraction()
        b = get_random_fraction()
        result = field_multiply(a, b)
        assert is_in_field_F(result), f"Multiplicative closure failed: {a} * {b} = {result} is not in F"

# 7. 乘法結合性
def test_multiplicative_associativity():
    for _ in range(NUM_TEST_CASES):
        a = get_random_fraction()
        b = get_random_fraction()
        c = get_random_fraction()
        assert field_multiply(field_multiply(a, b), c) == field_multiply(a, field_multiply(b, c)), \
            f"Multiplicative associativity failed: ({a} * {b}) * {c} != {a} * ({b} * {c})"

# 8. 乘法單位元素 (1 != 0)
def test_multiplicative_identity():
    assert MULTIPLICATIVE_IDENTITY != ADDITIVE_IDENTITY, "Multiplicative identity 1 should not be equal to additive identity 0"
    for _ in range(NUM_TEST_CASES):
        a = get_random_fraction()
        assert field_multiply(a, MULTIPLICATIVE_IDENTITY) == a, \
            f"Multiplicative identity failed (right): {a} * {MULTIPLICATIVE_IDENTITY} != {a}"
        assert field_multiply(MULTIPLICATIVE_IDENTITY, a) == a, \
            f"Multiplicative identity failed (left): {MULTIPLICATIVE_IDENTITY} * {a} != {a}"

# 9. 乘法反元素 (a != 0)
def test_multiplicative_inverse():
    for _ in range(NUM_TEST_CASES):
        a = get_random_nonzero_fraction() # 確保 a 不為 0
        a_inverse = get_multiplicative_inverse(a)
        assert is_in_field_F(a_inverse), f"Multiplicative inverse {a_inverse} for {a} is not in F"
        assert field_multiply(a, a_inverse) == MULTIPLICATIVE_IDENTITY, \
            f"Multiplicative inverse failed (right): {a} * {a_inverse} != {MULTIPLICATIVE_IDENTITY}"
        assert field_multiply(a_inverse, a) == MULTIPLICATIVE_IDENTITY, \
            f"Multiplicative inverse failed (left): {a_inverse} * {a} != {MULTIPLICATIVE_IDENTITY}"

# 10. 乘法交換性
def test_multiplicative_commutativity():
    for _ in range(NUM_TEST_CASES):
        a = get_random_fraction()
        b = get_random_fraction()
        assert field_multiply(a, b) == field_multiply(b, a), \
            f"Multiplicative commutativity failed: {a} * {b} != {b} * {a}"

# --- 第三組：兩種運算之間的關聯 ---

# 11. 分配律
def test_distributivity():
    for _ in range(NUM_TEST_CASES):
        a = get_random_fraction()
        b = get_random_fraction()
        c = get_random_fraction()
        # 左分配律
        assert field_multiply(a, field_add(b, c)) == field_add(field_multiply(a, b), field_multiply(a, c)), \
            f"Left distributivity failed: {a} * ({b} + {c}) != ({a} * {b}) + ({a} * {c})"
        # 右分配律 (因為乘法交換律，左分配律成立通常右分配律也成立，但驗證一下更完整)
        assert field_multiply(field_add(a, b), c) == field_add(field_multiply(a, c), field_multiply(b, c)), \
            f"Right distributivity failed: ({a} + {b}) * {c} != ({a} * {c}) + ({b} * {c})"