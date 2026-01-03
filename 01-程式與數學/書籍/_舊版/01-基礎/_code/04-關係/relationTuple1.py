students = {"Alice", "Bob", "Charlie"}
courses = {"Math", "Programming", "Physics"}
# 學生修習課程的關係 (定義域是學生，對應域是課程)

# 方法一：使用 set 儲存元組 (推薦，因為關係是集合)
enrollment_relation = {
    ("Alice", "Math"),
    ("Alice", "Programming"),
    ("Bob", "Physics"),
    ("Charlie", "Math"),
    ("Charlie", "Programming")
}

print("--- 學生修習課程的關係 (使用 set) ---")
print(enrollment_relation)

# 判斷某個配對是否存在於關係中
print("\nAlice 修了 Math 嗎？", ("Alice", "Math") in enrollment_relation)
print("Bob 修了 Programming 嗎？", ("Bob", "Programming") in enrollment_relation)

# 找出某個學生修了哪些課 (即定義域中某個元素的輸出)
def get_courses_for_student(student_name, relation):
    student_courses = set()
    for student, course in relation:
        if student == student_name:
            student_courses.add(course)
    return student_courses

print("\nAlice 修的課程:", get_courses_for_student("Alice", enrollment_relation))
print("Bob 修的課程:", get_courses_for_student("Bob", enrollment_relation))
print("David 修的課程:", get_courses_for_student("David", enrollment_relation)) # 沒有 David

# 判斷這個關係是否是「函數」
# (每個學生最多修一門課，且每個學生都修了一門課)
def is_function(relation, domain_set):
    # 檢查唯一性：每個輸入最多一個輸出
    outputs_count = {}
    for input_val, _ in relation:
        outputs_count[input_val] = outputs_count.get(input_val, 0) + 1
        if outputs_count[input_val] > 1:
            return False # 找到一個輸入有多個輸出

    # 檢查完全性：定義域中的每個元素都有輸出
    for domain_element in domain_set:
        if domain_element not in outputs_count:
            return False # 找到一個定義域元素沒有輸出
    
    return True

print("\n--- 檢查關係是否為函數 ---")
print("enrollment_relation 是函數嗎？", is_function(enrollment_relation, students)) # False (Alice 修了多門課)

# 創建一個「是函數」的關係：每個學生只修一門課
single_enrollment_function = {
    ("Alice", "Math"),
    ("Bob", "Physics"),
    ("Charlie", "Programming")
}
print("single_enrollment_function 是函數嗎？", is_function(single_enrollment_function, students)) # True
