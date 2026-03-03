# 使用 decimal 模組處理精確小數
from decimal import Decimal, getcontext

# 設定精確度，例如 20 位小數
getcontext().prec = 20

a = Decimal('0.1')
b = Decimal('0.2')
c = Decimal('0.3')

print("\n--- 使用 Decimal 模組 ---")
print("a + b:", a + b)
print("a + b == c?", (a + b) == c) # True