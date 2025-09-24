import random # 用來產生隨機測試數據

# 設定有限體的模數
MODULUS = 7 # 以 GF(7) 為例，您可根據需要修改此值，必須是質數，才會是有限體

op = '*'
identity = 1 # 乘法的單位元素

# 我們的集合是 {0, 1, ..., MODULUS - 1}
# 運算是模數乘法
def operation(a, b):
    """模擬有限體的乘法，結果取模數"""
    if not (include(a) and include(b)):
        raise ValueError("Input elements must be in the finite field.")
    return (a * b) % MODULUS

def extended_gcd(a, b):
    """使用擴展歐幾里得演算法來計算 (g, x, y) 使得 ax + by = g"""
    if a == 0:
        return (b, 0, 1)
    d, x1, y1 = extended_gcd(b % a, a)
    x = y1 - (b // a) * x1
    y = x1
    return (d, x, y)

def inverse(val):
    """計算乘法的反元素"""
    # 乘法的反元素只存在於非零元素
    if not include(val) or val == 0:
        raise ValueError("Multiplicative inverse only exists for non-zero elements in the finite field.")
    
    # 使用擴展歐幾里得演算法計算反元素
    # 我們需要找到 x 使得 val * x ≡ 1 (mod MODULUS)
    g, x, y = extended_gcd(val, MODULUS)
    
    # 如果 g 不是 1，代表 val 和 MODULUS 不互質，反元素不存在
    if g != 1:
        raise ValueError(f"Inverse does not exist for {val} in GF({MODULUS}).")
        
    return x % MODULUS

# 輔助函式，生成隨機的體內元素
def random_generate():
    """生成隨機的非零元素用於乘法測試"""
    # 乘法的反元素只存在於非零元素，所以我們生成 1 到 MODULUS-1 的隨機數
    return random.randint(1, MODULUS - 1)

def include(element):
    """檢查元素是否屬於我們的有限體集合 G"""
    return isinstance(element, int) and 0 <= element < MODULUS

# 測試用例
if __name__ == "__main__":
    print(f"--- 測試 GF({MODULUS}) 的乘法模組 ---")
    
    # 測試運算
    a = random_generate()
    b = random_generate()
    result = operation(a, b)
    print(f"乘法測試: {a} * {b} (mod {MODULUS}) = {result}")

    # 測試單位元素
    c = random_generate()
    result_identity = operation(c, identity)
    print(f"單位元素測試: {c} * {identity} (mod {MODULUS}) = {result_identity}")

    # 測試反元素
    d = random_generate()
    inv_d = inverse(d)
    result_inverse = operation(d, inv_d)
    print(f"反元素測試: {d} * {inv_d} (mod {MODULUS}) = {result_inverse} (應為 {identity})")
    
    # 測試特殊情況：0 的乘法
    print(f"0 的乘法測試: {random_generate()} * 0 (mod {MODULUS}) = {operation(random_generate(), 0)}")
