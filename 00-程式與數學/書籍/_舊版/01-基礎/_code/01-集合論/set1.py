# 創建一個集合
# 注意：用大括號 {} 來定義集合，但如果是空集合，要用 set() 而不是 {}，因為 {} 是用來定義空字典的
my_favorite_numbers = {1, 5, 8, 10}
even_numbers_small = {2, 4, 6, 8, 10}
prime_numbers_small = {2, 3, 5, 7}

print("我喜歡的數字:", my_favorite_numbers)
print("小於10的偶數:", even_numbers_small)
print("小於10的質數:", prime_numbers_small)

# 集合的無序性：即使順序不同，它們也是一樣的集合
set_a = {1, 2, 3}
set_b = {3, 1, 2}
print("\nset_a 和 set_b 是否相等？", set_a == set_b)

# 集合的互異性：重複的元素會被自動移除
numbers_with_duplicates = {1, 2, 2, 3, 4, 4, 4, 5}
print("有重複元素的集合:", numbers_with_duplicates) # 輸出 {1, 2, 3, 4, 5}

# 檢查元素是否在集合中 (判斷確定性)
print("5 在我的最愛數字裡嗎？", 5 in my_favorite_numbers)
print("7 在我的最愛數字裡嗎？", 7 in my_favorite_numbers)

# 集合的常用操作

# 1. 聯集 (Union): 合併兩個集合，移除重複元素
# 符號：A ∪ B
# 想像：把兩個袋子裡的東西都倒出來，重複的只算一次
all_numbers = my_favorite_numbers.union(even_numbers_small)
print("\n聯集 (我的最愛 U 偶數):", all_numbers)
# 也可以用 | 運算符
all_numbers_operator = my_favorite_numbers | even_numbers_small
print("聯集 (使用運算符):", all_numbers_operator)

# 2. 交集 (Intersection): 找出兩個集合中共同的元素
# 符號：A ∩ B
# 想像：兩個袋子裡都有的共同物品
common_numbers = my_favorite_numbers.intersection(even_numbers_small)
print("交集 (我的最愛 ∩ 偶數):", common_numbers)
# 也可以用 & 運算符
common_numbers_operator = my_favorite_numbers & even_numbers_small
print("交集 (使用運算符):", common_numbers_operator)

# 3. 差集 (Difference): 找出在第一個集合，但不在第二個集合的元素
# 符號：A - B
# 想像：從第一個袋子裡拿出第二個袋子裡也有的物品
my_unique_numbers = my_favorite_numbers.difference(even_numbers_small)
print("差集 (我的最愛 - 偶數):", my_unique_numbers) # 在我的最愛中，但不是偶數的
# 也可以用 - 運算符
my_unique_numbers_operator = my_favorite_numbers - even_numbers_small
print("差集 (使用運算符):", my_unique_numbers_operator)

# 4. 對稱差集 (Symmetric Difference): 找出只存在於其中一個集合的元素 (不重疊的)
# 符號：A Δ B
# 想像：把兩個袋子裡的東西都倒出來，但把兩者都有的重複物品丟掉
unique_to_either = my_favorite_numbers.symmetric_difference(even_numbers_small)
print("對稱差集 (我的最愛 Δ 偶數):", unique_to_either)
# 也可以用 ^ 運算符
unique_to_either_operator = my_favorite_numbers ^ even_numbers_small
print("對稱差集 (使用運算符):", unique_to_either_operator)

# 5. 子集 (Subset) 和 超集 (Superset)
# 符號：A ⊆ B (A 是 B 的子集), B ⊇ A (B 是 A 的超集)
# 想像：一個袋子裡的所有東西都在另一個袋子裡
set_c = {1, 5}
print("\n集合 c:", set_c)
print("c 是 my_favorite_numbers 的子集嗎？", set_c.issubset(my_favorite_numbers))
print("my_favorite_numbers 是 c 的超集嗎？", my_favorite_numbers.issuperset(set_c))

# 沒有共同元素的集合 (Disjoint sets)
# 想像：兩個袋子沒有任何相同的東西
set_d = {100, 200}
print("\n集合 d:", set_d)
print("my_favorite_numbers 和 d 是沒有共同元素的集合嗎？", my_favorite_numbers.isdisjoint(set_d))
print("my_favorite_numbers 和 even_numbers_small 是沒有共同元素的集合嗎？", my_favorite_numbers.isdisjoint(even_numbers_small)) # 這是 False，因為有共同元素 8, 10