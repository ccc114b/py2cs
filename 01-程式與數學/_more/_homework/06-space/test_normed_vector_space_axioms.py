# test_normed_vector_space_axioms.py
import pytest
import random
import math
from vector_space_elements import Vector2D, ZERO_VECTOR, is_vector_in_space, is_scalar

# 測試用的隨機純量和向量分量範圍
SCALAR_RANGE = 10
COMPONENT_RANGE = 10
NUM_TEST_CASES = 50

# 輔助函式，生成隨機向量
def get_random_vector():
    return Vector2D(random.uniform(-COMPONENT_RANGE, COMPONENT_RANGE),
                    random.uniform(-COMPONENT_RANGE, COMPONENT_RANGE))

# 輔助函式，生成隨機純量
def get_random_scalar():
    return random.uniform(-SCALAR_RANGE, SCALAR_RANGE)

# ----------------------------------------------------------------------
# 向量空間的八個公理 (這裡只列出部分關鍵的，更多可自行補充)
# ----------------------------------------------------------------------

# 1. 向量加法封閉性 (Closure under Addition)
def test_vector_addition_closure():
    for _ in range(NUM_TEST_CASES):
        u = get_random_vector()
        v = get_random_vector()
        result = u + v
        assert is_vector_in_space(result), f"Addition closure failed: {u} + {v} = {result} is not a Vector2D"

# 2. 向量加法結合性 (Associativity of Addition)
def test_vector_addition_associativity():
    for _ in range(NUM_TEST_CASES):
        u = get_random_vector()
        v = get_random_vector()
        w = get_random_vector()
        assert (u + v) + w == u + (v + w), \
            f"Addition associativity failed: ({u} + {v}) + {w} != {u} + ({v} + {w})"

# 3. 向量加法單位元素 (Additive Identity)
def test_vector_additive_identity():
    for _ in range(NUM_TEST_CASES):
        u = get_random_vector()
        assert (u + ZERO_VECTOR) == u, \
            f"Additive identity failed (right): {u} + {ZERO_VECTOR} != {u}"
        assert (ZERO_VECTOR + u) == u, \
            f"Additive identity failed (left): {ZERO_VECTOR} + {u} != {u}"

# 4. 向量加法反元素 (Additive Inverse)
def test_vector_additive_inverse():
    for _ in range(NUM_TEST_CASES):
        u = get_random_vector()
        u_inverse = u * -1 # 負向量
        assert (u + u_inverse) == ZERO_VECTOR, \
            f"Additive inverse failed: {u} + {u_inverse} != {ZERO_VECTOR}"

# 5. 純量乘法封閉性 (Closure under Scalar Multiplication)
def test_scalar_multiplication_closure():
    for _ in range(NUM_TEST_CASES):
        c = get_random_scalar()
        u = get_random_vector()
        result = c * u
        assert is_vector_in_space(result), f"Scalar multiplication closure failed: {c} * {u} = {result} is not a Vector2D"

# 6. 純量乘法結合性 (Associativity of Scalar Multiplication)
def test_scalar_multiplication_associativity():
    for _ in range(NUM_TEST_CASES):
        c = get_random_scalar()
        d = get_random_scalar()
        u = get_random_vector()
        assert (c * d) * u == c * (d * u), \
            f"Scalar multiplication associativity failed: ({c} * {d}) * {u} != {c} * ({d} * {u})"

# 7. 純量乘法單位元素 (Multiplicative Identity for Scalars)
def test_scalar_multiplication_identity():
    for _ in range(NUM_TEST_CASES):
        u = get_random_vector()
        assert (1 * u) == u, \
            f"Scalar multiplication identity failed: 1 * {u} != {u}"

# 8. 分配律 (Distributivity)
def test_distributivity_scalar_over_vector_addition():
    for _ in range(NUM_TEST_CASES):
        c = get_random_scalar()
        u = get_random_vector()
        v = get_random_vector()
        assert c * (u + v) == (c * u) + (c * v), \
            f"Distributivity failed: {c} * ({u} + {v}) != ({c} * {u}) + ({c} * {v})"

def test_distributivity_vector_over_scalar_addition():
    for _ in range(NUM_TEST_CASES):
        c = get_random_scalar()
        d = get_random_scalar()
        u = get_random_vector()
        assert (c + d) * u == (c * u) + (d * u), \
            f"Distributivity failed: ({c} + {d}) * {u} != ({c} * {u}) + ({d} * {u})"

# ----------------------------------------------------------------------
# 賦範向量空間的四個範數公理
# ----------------------------------------------------------------------

# 1. 非負性 (Non-negativity)
def test_norm_non_negativity():
    for _ in range(NUM_TEST_CASES):
        u = get_random_vector()
        assert u.norm() >= 0, f"Norm non-negativity failed: ||{u}|| = {u.norm()} < 0"

# 2. 正定性 (Positive-definiteness)
def test_norm_positive_definiteness():
    # 如果 norm 是 0，則向量必須是零向量
    assert ZERO_VECTOR.norm() == 0, f"Norm of zero vector is not 0: ||{ZERO_VECTOR}|| = {ZERO_VECTOR.norm()}"
    for _ in range(NUM_TEST_CASES):
        u = get_random_vector()
        if u.norm() == 0: # 考慮浮點數精度，用 isclose 比較
            assert u == ZERO_VECTOR, f"Norm is 0 but vector is not zero: {u}"
        else:
            assert u != ZERO_VECTOR, f"Norm is non-zero but vector is zero: {u}"

# 3. 純量乘法下的齊次性 (Homogeneity under Scalar Multiplication)
def test_norm_homogeneity():
    for _ in range(NUM_TEST_CASES):
        c = get_random_scalar()
        u = get_random_vector()
        assert math.isclose((c * u).norm(), abs(c) * u.norm()), \
            f"Norm homogeneity failed: ||{c} * {u}|| = {(c * u).norm()} != |{c}| * ||{u}|| = {abs(c) * u.norm()}"

# 4. 三角不等式 (Triangle Inequality)
def test_norm_triangle_inequality():
    for _ in range(NUM_TEST_CASES):
        u = get_random_vector()
        v = get_random_vector()
        # 由於浮點數精度，我們檢查左邊是否小於或約等於右邊
        assert (u + v).norm() <= u.norm() + v.norm() + 1e-9, \
            f"Triangle inequality failed: ||{u} + {v}|| = {(u + v).norm()} > ||{u}|| + ||{v}|| = {u.norm() + v.norm()}"

# ----------------------------------------------------------------------
# 完備性 (Completeness) - 無法直接測試，只能概念性理解
# ----------------------------------------------------------------------

# 這裡我們無法用 Pytest 直接驗證「完備性」，因為這涉及到無限序列的極限。
# 但我們可以理解，如果一個柯西序列 (元素越來越接近) 的極限點也在這個空間內，
# 那麼這個空間就是完備的。實數 R 和 R^n 在歐幾里得範數下都是完備的。

# 演示一個柯西序列的概念 (不作為 Pytest 測試)
def demonstrate_cauchy_sequence():
    print("\n--- Demonstrating a Cauchy Sequence (Conceptual) ---")
    # 考慮一個收斂到 (0.5, 0.5) 的序列
    sequence = []
    for i in range(1, 6):
        x = 0.5 - 1/(2**i)
        y = 0.5 - 1/(2**i)
        sequence.append(Vector2D(x, y))
        print(f"x_{i} = {sequence[-1]}")

    # 檢查序列中的點彼此越來越近
    for i in range(len(sequence) - 1):
        for j in range(i + 1, len(sequence)):
            dist = (sequence[i] - sequence[j]).norm()
            print(f"Distance between x_{i+1} and x_{j+1}: {dist:.6f}")
            # 你會看到距離隨著 i, j 變大而變小

    # 這個序列的極限是 Vector2D(0.5, 0.5)，它確實屬於 R^2 空間。
    # 因此，R^2 是完備的。

# 你可以在 main 函式或單獨的腳本中呼叫這個演示函式
if __name__ == "__main__":
    pytest.main(['-v', 'test_normed_vector_space_axioms.py'])
    demonstrate_cauchy_sequence()