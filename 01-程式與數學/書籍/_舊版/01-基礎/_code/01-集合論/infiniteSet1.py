def natural_numbers_generator():
    n = 0
    while True:
        yield n  # 產生下一個自然數
        n += 1

# 我們不能全部列出來，但可以取出前幾個
print("--- 模擬自然數集合 (N) ---")
nat_gen = natural_numbers_generator()
print("前5個自然數:")
for _ in range(5):
    print(next(nat_gen))

# 你可以一直要下去，但它不會一次性生成
# print(next(nat_gen)) # 會是 5
# print(next(nat_gen)) # 會是 6