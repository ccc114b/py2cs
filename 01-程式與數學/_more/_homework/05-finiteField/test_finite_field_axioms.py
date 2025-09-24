# test_finite_field_axioms.py
import pytest
import random
from finite_field_elements import (
    FiniteFieldElement,
    get_additive_identity, get_multiplicative_identity,
    is_in_field_F
)

MODULUS = 5 # 我們在 GF(5) 上進行測試
NUM_TEST_CASES = 50

# 輔助函式，生成 GF(MODULUS) 中的隨機元素
def get_random_field_element():
    return FiniteFieldElement(random.randint(0, MODULUS - 1), MODULUS)

def get_random_nonzero_field_element():
    while True:
        e = get_random_field_element()
        if e != get_additive_identity(MODULUS):
            return e

# 定義加法和乘法的單位元素
ADDITIVE_IDENTITY = get_additive_identity(MODULUS)
MULTIPLICATIVE_IDENTITY = get_multiplicative_identity(MODULUS)

# --- 第一組：關於加法，形成一個交換群 (F, +) ---

# 1. 加法封閉性
def test_additive_closure_gf():
    for _ in range(NUM_TEST_CASES):
        a = get_random_field_element()
        b = get_random_field_element()
        result = a + b
        assert is_in_field_F(result, MODULUS), f"Additive closure failed: {a} + {b} = {result} is not in GF({MODULUS})"

# 2. 加法結合性
def test_additive_associativity_gf():
    for _ in range(NUM_TEST_CASES):
        a = get_random_field_element()
        b = get_random_field_element()
        c = get_random_field_element()
        assert (a + b) + c == a + (b + c), \
            f"Additive associativity failed: ({a} + {b}) + {c} != {a} + ({b} + {c})"

# 3. 加法單位元素
def test_additive_identity_gf():
    for _ in range(NUM_TEST_CASES):
        a = get_random_field_element()
        assert (a + ADDITIVE_IDENTITY) == a, \
            f"Additive identity failed (right): {a} + {ADDITIVE_IDENTITY} != {a}"
        assert (ADDITIVE_IDENTITY + a) == a, \
            f"Additive identity failed (left): {ADDITIVE_IDENTITY} + {a} != {a}"

# 4. 加法反元素
def test_additive_inverse_gf():
    for _ in range(NUM_TEST_CASES):
        a = get_random_field_element()
        a_inverse = -a
        assert is_in_field_F(a_inverse, MODULUS), f"Additive inverse {a_inverse} for {a} is not in GF({MODULUS})"
        assert (a + a_inverse) == ADDITIVE_IDENTITY, \
            f"Additive inverse failed (right): {a} + {a_inverse} != {ADDITIVE_IDENTITY}"
        assert (a_inverse + a) == ADDITIVE_IDENTITY, \
            f"Additive inverse failed (left): {a_inverse} + {a} != {ADDITIVE_IDENTITY}"

# 5. 加法交換性
def test_additive_commutativity_gf():
    for _ in range(NUM_TEST_CASES):
        a = get_random_field_element()
        b = get_random_field_element()
        assert (a + b) == (b + a), \
            f"Additive commutativity failed: {a} + {b} != {b} + {a}"

# --- 第二組：關於乘法，形成一個非零元素的交換群 (F \ {0}, x) ---

# 6. 乘法封閉性
def test_multiplicative_closure_gf():
    for _ in range(NUM_TEST_CASES):
        a = get_random_field_element()
        b = get_random_field_element()
        result = a * b
        assert is_in_field_F(result, MODULUS), f"Multiplicative closure failed: {a} * {b} = {result} is not in GF({MODULUS})"

# 7. 乘法結合性
def test_multiplicative_associativity_gf():
    for _ in range(NUM_TEST_CASES):
        a = get_random_field_element()
        b = get_random_field_element()
        c = get_random_field_element()
        assert (a * b) * c == a * (b * c), \
            f"Multiplicative associativity failed: ({a} * {b}) * {c} != {a} * ({b} * {c})"

# 8. 乘法單位元素 (1 != 0)
def test_multiplicative_identity_gf():
    assert MULTIPLICATIVE_IDENTITY != ADDITIVE_IDENTITY, "Multiplicative identity 1 should not be equal to additive identity 0"
    for _ in range(NUM_TEST_CASES):
        a = get_random_field_element()
        assert (a * MULTIPLICATIVE_IDENTITY) == a, \
            f"Multiplicative identity failed (right): {a} * {MULTIPLICATIVE_IDENTITY} != {a}"
        assert (MULTIPLICATIVE_IDENTITY * a) == a, \
            f"Multiplicative identity failed (left): {MULTIPLICATIVE_IDENTITY} * {a} != {a}"

# 9. 乘法反元素 (a != 0)
def test_multiplicative_inverse_gf():
    for _ in range(NUM_TEST_CASES):
        a = get_random_nonzero_field_element() # 確保 a 不為 0
        a_inverse = a.multiplicative_inverse()
        assert is_in_field_F(a_inverse, MODULUS), f"Multiplicative inverse {a_inverse} for {a} is not in GF({MODULUS})"
        assert (a * a_inverse) == MULTIPLICATIVE_IDENTITY, \
            f"Multiplicative inverse failed (right): {a} * {a_inverse} != {MULTIPLICATIVE_IDENTITY}"
        assert (a_inverse * a) == MULTIPLICATIVE_IDENTITY, \
            f"Multiplicative inverse failed (left): {a_inverse} * {a} != {MULTIPLICATIVE_IDENTITY}"

# 10. 乘法交換性
def test_multiplicative_commutativity_gf():
    for _ in range(NUM_TEST_CASES):
        a = get_random_field_element()
        b = get_random_field_element()
        assert (a * b) == (b * a), \
            f"Multiplicative commutativity failed: {a} * {b} != {b} * {a}"

# --- 第三組：兩種運算之間的關聯 ---

# 11. 分配律
def test_distributivity_gf():
    for _ in range(NUM_TEST_CASES):
        a = get_random_field_element()
        b = get_random_field_element()
        c = get_random_field_element()
        # 左分配律
        assert a * (b + c) == (a * b) + (a * c), \
            f"Left distributivity failed: {a} * ({b} + {c}) != ({a} * {b}) + ({a} * {c})"
        # 右分配律 (因為乘法交換律，左分配律成立通常右分配律也成立，但驗證一下更完整)
        assert (a + b) * c == (a * c) + (b * c), \
            f"Right distributivity failed: ({a} + {b}) * {c} != ({a} * {c}) + ({b} * {c})"