Y = lambda f:\
  (lambda x:f(lambda y:x(x)(y)))\
  (lambda x:f(lambda y:x(x)(y)))
  
# 遞迴生成器函數 (f)
# 這個函數接收一個參數 rec，代表遞迴呼叫
# 接著再接收 n，代表我們想計算的數字
def factorial_generator(rec):
    return lambda n: 1 if n == 0 else n * rec(n - 1)

# 使用 Y-Combinator 產生可遞迴的階乘函數
# 我們將 factorial_generator 傳入 Y，Y 會返回一個可以遞迴的函數
factorial = Y(factorial_generator)

# 測試階乘函數
print(factorial(5))  # 預期輸出：120
print(factorial(10)) # 預期輸出：3628800