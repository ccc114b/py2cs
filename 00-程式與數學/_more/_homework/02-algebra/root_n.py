import numpy as np

def polynomial_value(c, z):
    """
    計算多項式 P(z) 的值。
    c: 係數陣列，c[i] 是 x^i 的係數。
    z: 複數輸入。
    """
    val = 0
    for i, coeff in enumerate(c):
        val += coeff * (z**i)
    return val

def polynomial_derivative_value(c, z):
    """
    計算多項式 P'(z) 的值。
    c: 係數陣列。
    z: 複數輸入。
    """
    derivative_coeffs = []
    for i in range(1, len(c)):
        derivative_coeffs.append(i * c[i])
    
    if not derivative_coeffs: # 如果是常數多項式，導數為 0
        return 0
    
    return polynomial_value(derivative_coeffs, z)

def find_roots_gradient_descent(c, initial_guesses, learning_rate=0.01, num_iterations=10000, tolerance=1e-8):
    """
    使用梯度下降法尋找多項式的根。
    c: 係數陣列，c[i] 是 x^i 的係數。
    initial_guesses: 複數的初始猜測陣列。
    learning_rate: 學習率。
    num_iterations: 迭代次數。
    tolerance: 收斂容忍度。
    """
    
    roots_found = []
    
    # 對每個初始猜測進行梯度下降
    for initial_z in initial_guesses:
        z = initial_z
        
        for i in range(num_iterations):
            p_z = polynomial_value(c, z)
            
            # 如果已經足夠接近根，則停止
            if abs(p_z) < tolerance:
                break
            
            p_prime_z = polynomial_derivative_value(c, z)
            
            # 避免除以零或梯度過小導致停滯（雖然這裡不是直接除以零）
            # 如果導數為零，梯度為零，會停留在原地
            if p_prime_z == 0:
                # 在導數為零的地方可能會有問題，例如局部極小值
                # 這裡簡單地跳出，或者可以添加一個小的隨機擾動
                break 
            
            # 梯度下降更新規則
            # L(z) = P(z) * conj(P(z))
            # dL/d(conj(z)) = P(z) * conj(P'(z))
            # z_new = z_old - eta * dL/d(conj(z))
            gradient_term = p_z * np.conj(p_prime_z)
            z -= learning_rate * gradient_term
            
        # 檢查找到的根是否已經存在（考慮浮點數誤差）
        is_new_root = True
        for existing_root in roots_found:
            if abs(z - existing_root) < tolerance * 10: # 稍微放寬一點誤差
                is_new_root = False
                break
        
        if is_new_root and abs(polynomial_value(c, z)) < tolerance:
            roots_found.append(z)
            
    return roots_found

# --- 範例使用 ---

# 1. 一元二次多項式: x^2 - 1 = 0  => 根為 1, -1
# 係數陣列: c[0] + c[1]x + c[2]x^2
coeffs1 = np.array([-1, 0, 1], dtype=complex) 
initial_guesses1 = [0.5 + 0.5j, -0.5 - 0.5j, 10 + 10j, -10 - 10j] # 提供多個初始猜測
roots1 = find_roots_gradient_descent(coeffs1, initial_guesses1, learning_rate=0.001, num_iterations=20000)
print(f"多項式 x^2 - 1 的根: {roots1}")
# 預期輸出接近 [1, -1]

# 2. 一元三次多項式: x^3 - 6x^2 + 11x - 6 = 0 => 根為 1, 2, 3
coeffs2 = np.array([-6, 11, -6, 1], dtype=complex)
initial_guesses2 = [0.5 + 0.5j, 1.5 - 0.5j, 2.5 + 0.5j, 0, 10]
roots2 = find_roots_gradient_descent(coeffs2, initial_guesses2, learning_rate=0.0001, num_iterations=50000)
print(f"多項式 x^3 - 6x^2 + 11x - 6 的根: {roots2}")
# 預期輸出接近 [1, 2, 3]

# 3. 帶有複數根的多項式: x^2 + 1 = 0 => 根為 i, -i
coeffs3 = np.array([1, 0, 1], dtype=complex)
initial_guesses3 = [0.1 + 0.1j, -0.1 - 0.1j, 1+1j, -1-1j]
roots3 = find_roots_gradient_descent(coeffs3, initial_guesses3, learning_rate=0.001, num_iterations=20000)
print(f"多項式 x^2 + 1 的根: {roots3}")
# 預期輸出接近 [i, -i]

# 4. 常數多項式 (沒有根，或者說所有複數都不是根，除非 c[0]=0)
coeffs4 = np.array([5], dtype=complex) # P(z) = 5
initial_guesses4 = [0]
roots4 = find_roots_gradient_descent(coeffs4, initial_guesses4)
print(f"多項式 5 的根: {roots4}") # 預期輸出為 [] (因為沒有根)

coeffs5 = np.array([0], dtype=complex) # P(z) = 0
initial_guesses5 = [0]
roots5 = find_roots_gradient_descent(coeffs5, initial_guesses5)
print(f"多項式 0 的根: {roots5}") # 預期輸出為 [] (因為定義上來說，所有複數都是根，但梯度下降不會收斂到一個特定根)