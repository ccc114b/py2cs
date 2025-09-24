from cmath import sin, cos, exp, pi

def euler_formula(theta):
    """
    使用歐拉公式計算 e^(i*theta)。
    e^(i*theta) = cos(theta) + i*sin(theta)
    """
    return cos(theta) + 1j * sin(theta)

# 測試歐拉公式
if __name__ == "__main__":
    angles = [0, pi/6, pi/4, pi/2, pi, 3*pi/2, 2*pi]
    for angle in angles:
        result = euler_formula(angle)
        print(f"e^(i*{angle}) = {result} (預期: {exp(1j*angle)})")
        assert abs(result - exp(1j*angle)) < 1e-10, "測試失敗！"
    print("所有測試通過！")
