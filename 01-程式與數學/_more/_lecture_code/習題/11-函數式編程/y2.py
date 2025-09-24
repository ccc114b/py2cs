# Y 組合子的 Python 實現
Y = lambda f:\
  (lambda x:f(lambda y:x(x)(y)))\
  (lambda x:f(lambda y:x(x)(y)))

# 階乘的生成器函數
# 這個函數 f 負責接收遞迴調用
# 這個函數 n 才是我們計算階乘的數字
fact_generator = lambda f: lambda n: 1 if n == 0 else n * f(n - 1)


result = Y(fact_generator)(5)
print(result)