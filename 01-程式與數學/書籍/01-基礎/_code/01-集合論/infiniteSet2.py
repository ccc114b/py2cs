class InfiniteEvenSet:
    def __contains__(self, item):
        """
        定義 'in' 操作符的行為
        判斷一個元素是否在這個（無窮）集合中
        """
        if isinstance(item, int) and item % 2 == 0:
            return True
        return False

    def __str__(self):
        return "{..., -4, -2, 0, 2, 4, ...} (所有偶數的集合)"

    def get_generator(self, start=0):
        """
        提供一個生成器來遍歷部分元素
        """
        current = start if start % 2 == 0 else (start + 1 if start > 0 else 0)
        while True:
            yield current
            current += 2


even_set = InfiniteEvenSet()
print("\n--- 抽象的偶數集合類 ---")
print(even_set)

print("100 in even_set?", 100 in even_set)
print("99 in even_set?", 99 in even_set)
print("0 in even_set?", 0 in even_set)
print("Hello in even_set?", "Hello" in even_set)

print("\n從這個集合中取出前5個元素 (從 0 開始):")
even_gen_from_class = even_set.get_generator()
for _ in range(5):
    print(next(even_gen_from_class))

print("\n從這個集合中取出從 7 開始的前3個偶數:")
even_gen_from_class_start_7 = even_set.get_generator(start=7)
for _ in range(3):
    print(next(even_gen_from_class_start_7))