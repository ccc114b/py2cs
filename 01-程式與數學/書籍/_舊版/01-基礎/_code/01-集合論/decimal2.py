from decimal import Decimal, getcontext
import math # 引入 math 模組來獲取 math.pi 和 math.e

getcontext().prec = 5 # 設置精度為 5 位小數

# 獲取 math.pi 的值，它本身就是一個 float（雙精度浮點數）的近似值
# 然後將這個近似值轉換為 Decimal
pi_float_approx = math.pi
pi_decimal_approx_5_prec = Decimal(str(pi_float_approx))

print(f"math.pi (float): {pi_float_approx}")
print(f"Decimal('math.pi') (精度5): {pi_decimal_approx_5_prec}") # 會根據精度截斷或四捨五入

getcontext().prec = 15 # 設置精度為 15 位小數
pi_decimal_approx_15_prec = Decimal(str(pi_float_approx))
print(f"Decimal('math.pi') (精度15): {pi_decimal_approx_15_prec}")

getcontext().prec = 50 # 設置更高的精度
pi_decimal_approx_50_prec = Decimal(str(pi_float_approx))
print(f"Decimal('math.pi') (精度50): {pi_decimal_approx_50_prec}")
# 注意：即使設定了高精度，由於 math.pi 本身是 float，其精度是有限的，
# 所以轉換過來後，能提供的有效位數也是有限的。
# 如果需要更高精度的 pi，你需要從其他來源獲取更高精度的字串。

# 另一種方式：手動輸入高精度的無理數字串
high_prec_pi_str = "3.14159265358979323846264338327950288419716939937510"
pi_decimal_from_str = Decimal(high_prec_pi_str)

getcontext().prec = 10
print(f"\n從高精度字串創建Decimal(精度10): {pi_decimal_from_str}") # 會根據當前精度四捨五入
getcontext().prec = 30
print(f"從高精度字串創建Decimal(精度30): {pi_decimal_from_str}") # 會根據當前精度四捨五入

# 進行運算時，也是在設定的精度下進行
two = Decimal('2')
area = pi_decimal_from_str * two # 假設半徑是1，求直徑 (僅為示範)
print(f"pi * 2 (精度30): {area}")