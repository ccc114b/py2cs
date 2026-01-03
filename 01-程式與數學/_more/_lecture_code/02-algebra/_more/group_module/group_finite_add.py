import random # 用來產生隨機測試數據

# 設定有限體的模數
MODULUS = 7 # 以 GF(7) 為例，您可根據需要修改此值

op = '+'
identity = 0 # 加法的單位元素

# 我們的集合是 {0, 1, ..., MODULUS - 1}
# 運算是模數加法
def operation(a, b):
    """模擬有限體的加法，結果取模數"""
    if not (include(a) and include(b)):
        raise ValueError("Input elements must be in the finite field.")
    return (a + b) % MODULUS

def inverse(val):
    """計算加法的反元素"""
    if not include(val):
        raise ValueError("Input element must be in the finite field.")
    # (a + x) % p = 0，則 x = (-a) % p
    return (-val) % MODULUS

TEST_RANGE = MODULUS
# 輔助函式，生成隨機的體內元素
def random_generate():
    return random.randint(0, TEST_RANGE - 1)

def include(element):
    """檢查元素是否屬於我們的有限體集合 G"""
    return isinstance(element, int) and 0 <= element < MODULUS

# 測試用例
if __name__ == "__main__":
    print(f"--- 測試 GF({MODULUS}) 的加法模組 ---")
    
    # 測試運算
    a = random_generate()
    b = random_generate()
    result = operation(a, b)
    print(f"加法測試: {a} + {b} (mod {MODULUS}) = {result}")

    # 測試單位元素
    c = random_generate()
    result_identity = operation(c, identity)
    print(f"單位元素測試: {c} + {identity} (mod {MODULUS}) = {result_identity}")

    # 測試反元素
    d = random_generate()
    inv_d = inverse(d)
    result_inverse = operation(d, inv_d)
    print(f"反元素測試: {d} + {inv_d} (mod {MODULUS}) = {result_inverse} (應為 {identity})")

    # 測試邊界值
    e = MODULUS - 1
    f = 1
    result_boundary = operation(e, f)
    print(f"邊界值測試: {e} + {f} (mod {MODULUS}) = {result_boundary}")
