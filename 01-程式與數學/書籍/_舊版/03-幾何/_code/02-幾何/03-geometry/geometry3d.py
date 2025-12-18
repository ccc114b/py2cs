from vector import *
from math import *

def cross3d(a, b):
    """
    計算兩個三維向量的外積 (Cross Product)。

    參數:
    a (list or tuple): 第一個三維向量，例如 [a1, a2, a3]
    b (list or tuple): 第二個三維向量，例如 [b1, b2, b3]

    返回:
    list: 結果為一個新的三維向量 [c1, c2, c3]

    Raises:
    ValueError: 如果輸入的向量不是三維，會拋出錯誤。
    """
    # 檢查輸入向量的維度
    if len(a) != 3 or len(b) != 3:
        raise ValueError("Cross product is defined only for 3-dimensional vectors.")

    # 根據數學公式計算外積的三個分量
    c1 = a[1] * b[2] - a[2] * b[1]
    c2 = a[2] * b[0] - a[0] * b[2]
    c3 = a[0] * b[1] - a[1] * b[0]

    return [c1, c2, c3]

def normal(a): # 向量 a 的法向量
	return 

def rotate(a, rad): # 向量 a 旋轉 rad 角度
	return 

def pointOnPlane(p, plane): # 點 p 是否在平面 plane 上
    pass

def parallel(plane1, plane2): # 兩平面是否平行
    pass

def vertical(plane1, plane2): # 兩平面是否垂直
    pass

def lineCrossPlane(line, plane): # 直線與平面的交點
    pass

def volume4(a,b,c,d): # a,b,c,d 形成的四面體體積
    return dot(cross(sub(b,a), sub(c,a)), sub(d,a))/6.0

if __name__ == '__main__':

    # 測試我們的 cross 函數
    print("--- 外積 (Cross Product) 範例 ---")

    vector1 = [1, 0, 0] # x 軸單位向量
    vector2 = [0, 1, 0] # y 軸單位向量

    # 預期結果應該是 [0, 0, 1] (z 軸單位向量)
    result_cross = cross3d(vector1, vector2)
    print(f"向量 {vector1} 和 {vector2} 的外積是: {result_cross}") # 預期: [0, 0, 1]

    vector_a = [3, -3, 1]
    vector_b = [4, 9, 2]
    result_cross_ab = cross3d(vector_a, vector_b)
    print(f"向量 {vector_a} 和 {vector_b} 的外積是: {result_cross_ab}") # 預期: [-15, -2, 39]

    vector_p = [1, 2, 3]
    vector_q = [4, 5, 6]
    result_cross_pq = cross3d(vector_p, vector_q)
    print(f"向量 {vector_p} 和 {vector_q} 的外積是: {result_cross_pq}") # 預期: [-3, 6, -3]

    # 測試非三維向量的錯誤處理
    try:
        cross3d([1, 2], [3, 4, 5])
    except ValueError as e:
        print(f"錯誤測試成功: {e}")

    try:
        cross3d([1, 2, 3, 4], [5, 6, 7, 8])
    except ValueError as e:
        print(f"錯誤測試成功: {e}")
