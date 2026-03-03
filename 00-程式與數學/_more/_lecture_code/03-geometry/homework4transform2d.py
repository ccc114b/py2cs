import numpy as np
import math

def translation_matrix(tx, ty):
    """
    建立 2D 平移矩陣。

    Args:
        tx (float): x 方向的平移量。
        ty (float): y 方向的平移量。

    Returns:
        np.array: 3x3 平移矩陣。
    """
    return np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ], dtype=float)

def rotation_matrix(angle_rad):
    """
    建立 2D 旋轉矩陣。

    Args:
        angle_rad (float): 旋轉角度，單位為弧度。

    Returns:
        np.array: 3x3 旋轉矩陣。
    """
    cos_theta = math.cos(angle_rad)
    sin_theta = math.sin(angle_rad)
    return np.array([
        [cos_theta, -sin_theta, 0],
        [sin_theta, cos_theta, 0],
        [0, 0, 1]
    ], dtype=float)

def scaling_matrix(sx, sy):
    """
    建立 2D 縮放矩陣。

    Args:
        sx (float): x 方向的縮放倍率。
        sy (float): y 方向的縮放倍率。

    Returns:
        np.array: 3x3 縮放矩陣。
    """
    return np.array([
        [sx, 0, 0],
        [0, sy, 0],
        [0, 0, 1]
    ], dtype=float)

import numpy as np
import math

# (上述的 translation_matrix, rotation_matrix, scaling_matrix 函數放在這裡)
def translation_matrix(tx, ty):
    return np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=float)

def rotation_matrix(angle_rad):
    cos_theta = math.cos(angle_rad)
    sin_theta = math.sin(angle_rad)
    return np.array([[cos_theta, -sin_theta, 0], [sin_theta, cos_theta, 0], [0, 0, 1]], dtype=float)

def scaling_matrix(sx, sy):
    return np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=float)


# --- 測試範例 ---
if __name__ == "__main__":
    # 1. 將 2D 點轉換為齊次座標向量
    point_2d = np.array([10, 5])
    point_3d = np.append(point_2d, 1)

    # 2. 建立轉換矩陣
    # 旋轉 90 度 (π/2 弧度)
    rotation_mat = rotation_matrix(math.pi / 2)
    # 縮放 2 倍
    scaling_mat = scaling_matrix(2, 2)
    # 平移 (30, 40)
    translation_mat = translation_matrix(30, 40)

    # 3. 組合多個轉換
    # 轉換順序很重要！矩陣乘法不滿足交換律。
    # 這裡的順序是：先旋轉 -> 再縮放 -> 最後平移
    # 矩陣相乘時，轉換的應用順序是從右到左。
    # 因此，變換矩陣的組合是 T * S * R
    combined_transform = translation_mat @ scaling_mat @ rotation_mat

    # 4. 應用轉換
    transformed_point_3d = combined_transform @ point_3d

    # 5. 將結果從齊次座標變回 2D 點
    transformed_point_2d = transformed_point_3d[:2]

    print(f"原始 2D 點: {point_2d}")
    print(f"組合轉換矩陣:\n{combined_transform}\n")
    print(f"轉換後的 2D 點: {np.round(transformed_point_2d, 2)}")
    # 預期結果: 點(10, 5) 旋轉 90 度變成 (-5, 10)，然後縮放 2 倍變成 (-10, 20)，最後平移 (30, 40) 變成 (20, 60)。
