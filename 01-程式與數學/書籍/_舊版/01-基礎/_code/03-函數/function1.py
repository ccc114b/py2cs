# 類似數學函數：沒有副作用，純粹根據輸入計算輸出
def add_one(x):
    return x + 1

result = add_one(5)
print(f"add_one(5) = {result}") # 輸出 6

def power(a, n):
    return a ** n

print(f"power(2, 3) = {power(2, 3)}") # 輸出 8

def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)

print(f"factorial(5) = {factorial(5)}") # 輸出 120

# 具有副作用的函數：改變外部狀態，沒有明確返回值 (隱式返回 None)
global_counter = 0

def increment_counter():
    global global_counter # 聲明要修改全域變數
    global_counter += 1
    print(f"Counter incremented to {global_counter}")

print(f"Initial counter: {global_counter}") # 輸出 0
increment_counter() # 輸出 "Counter incremented to 1"
print(f"After first call: {global_counter}") # 輸出 1
increment_counter() # 輸出 "Counter incremented to 2"
print(f"After second call: {global_counter}") # 輸出 2

# 可能不完全的函數：在特定輸入下會出錯
def safe_divide(numerator, denominator):
    if denominator == 0:
        print("錯誤：除數不能為零！")
        return None # 返回 None 或拋出異常
    return numerator / denominator

print(f"safe_divide(10, 2) = {safe_divide(10, 2)}") # 輸出 5.0
print(f"safe_divide(10, 0) = {safe_divide(10, 0)}") # 輸出錯誤信息和 None