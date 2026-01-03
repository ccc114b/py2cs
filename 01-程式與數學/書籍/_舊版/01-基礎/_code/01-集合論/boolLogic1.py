print("--- 布林邏輯基本運算 ---")

# 假設陳述 P 為真，Q 為假
P = True
Q = False

# AND 運算 (只有 P 和 Q 都為真時才為真)
# P 且 Q
print(f"P and Q: {P and Q}") # False

# OR 運算 (P 或 Q 至少有一個為真時就為真)
# P 或 Q
print(f"P or Q: {P or Q}")   # True

# NOT 運算 (取反)
# 非 P
print(f"not P: {not P}")     # False
# 非 Q
print(f"not Q: {not Q}")     # True

# 複雜一點的組合
R = True
S = True
print(f"\n(P and R) or (not Q): {(P and R) or (not Q)}")
# (True and True) or (not False)
# True or True
# True

print(f"not (S or Q): {not (S or Q)}")
# not (True or False)
# not True
# False

# 條件判斷中的應用
age = 20
is_student = True

if age >= 18 and is_student:
    print("\n你可以獲得學生折扣！")
else:
    print("\n抱歉，不能獲得學生折扣。")

# 檢查一個數字是否在 0 到 10 之間 (且不等於 5)
num = 7
if num > 0 and num < 10 and num != 5:
    print(f"{num} 在 0 到 10 之間且不等於 5。")
else:
    print(f"{num} 不符合條件。")