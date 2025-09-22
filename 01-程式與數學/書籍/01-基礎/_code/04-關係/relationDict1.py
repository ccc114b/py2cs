print("\n--- 使用 dict 實作函數 ---")
# 學生和他們最喜歡的顏色 (假設每個學生只喜歡一種)
favorite_color_function = {
    "Alice": "Blue",
    "Bob": "Green",
    "Charlie": "Red"
}

print(f"Alice 喜歡的顏色是: {favorite_color_function['Alice']}")

# 字典自動保證了唯一性：如果你嘗試為同一個鍵賦值兩次，後面的會覆蓋前面的
favorite_color_function["Alice"] = "Yellow"
print(f"現在 Alice 喜歡的顏色是: {favorite_color_function['Alice']}") # 輸出 Yellow

# 檢查完全性 (如果需要的話)
all_students = {"Alice", "Bob", "Charlie", "David"}
is_total = all(student in favorite_color_function for student in all_students)
print(f"所有學生都有喜歡的顏色嗎？: {is_total}") # False (因為 David 沒有)