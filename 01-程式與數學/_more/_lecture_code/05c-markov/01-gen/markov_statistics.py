import numpy as np

def get_transition_matrix(sequence):
    """
    從序列中估計轉移矩陣。
    """
    # 1. 定義狀態空間
    states = sorted(list(set(sequence)))
    state_to_index = {state: i for i, state in enumerate(states)}
    n = len(states)

    # 2. 建立頻率計數矩陣
    counts = np.zeros((n, n))

    # 3. 遍歷序列並計數轉移
    for i in range(len(sequence) - 1):
        from_state = sequence[i]
        to_state = sequence[i+1]
        from_index = state_to_index[from_state]
        to_index = state_to_index[to_state]
        counts[from_index, to_index] += 1

    # 4. 正規化以計算機率
    # 避免除以零的錯誤
    row_sums = counts.sum(axis=1, keepdims=True)
    # 將所有行總和為零的元素替換為 1，以避免警告或錯誤
    row_sums[row_sums == 0] = 1
    
    transition_matrix = counts / row_sums

    return transition_matrix, states

# 範例序列
my_sequence = ['A', 'C', 'B', 'B', 'B', 'B', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'C', 'B', 'B', 'B', 'B', 'C', 'C', 'B', 'B', 'B', 'B', 'B', 'C', 'B', 'B', 'A', 'A', 'A', 'B', 'B', 'C', 'B', 'B', 'B', 'C', 'C', 'C', 'C', 'C', 'B', 'A', 'B', 'A', 'A', 'A', 'A', 'A', 'C', 'C', 'C', 'B', 'B', 'B', 'C', 'C', 'B', 'C', 'C', 'C', 'B', 'B', 'B', 'B', 'A', 'A', 'B', 'A', 'A', 'A', 'C', 'C', 'B', 'B', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'C', 'C', 'C', 'B', 'B', 'A', 'A', 'A', 'B', 'B', 'B', 'C', 'B', 'B', 'A', 'B', 'C', 'B', 'B', 'B', 'C', 'A', 'A', 'A', 'A', 'A', 'B', 'A', 'A', 'A', 'B', 'A', 'C', 'C', 'C', 'C', 'B', 'A', 'C', 'B', 'B', 'A', 'A', 'A', 'B', 'B', 'C', 'C', 'C', 'B', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'B', 'C', 'B', 'A', 'A', 'A', 'A', 'A', 'A', 'B', 'C', 'B', 'A', 'A', 'C', 'C', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'B', 'C', 'B', 'B', 'C', 'C', 'B', 'B', 'A', 'A', 'A', 'A', 'B', 'B', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B', 'B', 'C', 'C', 'A', 'A', 'B', 'B', 'B', 'A', 'A', 'A', 'B', 'A', 'A', 'B', 'A', 'B', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'C', 'C', 'C', 'C', 'C', 'C', 'B', 'B', 'C', 'C', 'C', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'B', 'C', 'B', 'A', 'A', 'A', 'A', 'B', 'B', 'C', 'B', 'A', 'B', 'B', 'C', 'C', 'A', 'A', 'A', 'C', 'A', 'A', 'A', 'A', 'B', 'B', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'A', 'A', 'A', 'C', 'C', 'C', 'B', 'B', 'B', 'B', 'C', 'B', 'A', 'A', 'B', 'A', 'A', 'A', 'C', 'B', 'A', 'C', 'A', 'A', 'C', 'C', 'B', 'B', 'B', 'C', 'B', 'A', 'A', 'A', 'B', 'A', 'A', 'C', 'B', 'A', 'A', 'B', 'A', 'B', 'C', 'B', 'A', 'A', 'B', 'B', 'A', 'B', 'B', 'A', 'A', 'B', 'A', 'A', 'A', 'A', 'B', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'B', 'C', 'B', 'C', 'A', 'B', 'B', 'B', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'C', 'B', 'C', 'C', 'B', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'A', 'A', 'B', 'B', 'A', 'A', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'C', 'C', 'B', 'A', 'A', 'A', 'A', 'A', 'A', 'B', 'A', 'A', 'A', 'A', 'A', 'A', 'B', 'A', 'C', 'B', 'A', 'A', 'A', 'C', 'B', 'B', 'B', 'C', 'C', 'B', 'C', 'B', 'B', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'C', 'A', 'A', 'A', 'A', 'A', 'C', 'B', 'B', 'B', 'A', 'B', 'B', 'B', 'B', 'B', 'B', 'A', 'C', 'C', 'B', 'C', 'B', 'A', 'A', 'C', 'C', 'B', 'B', 'A', 'B', 'B', 'A', 'A', 'A', 'A', 'A', 'B', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'C', 'B', 'B', 'A', 'A', 'B', 'C', 'B', 'C', 'B', 'A', 'A', 'B', 'B', 'C', 'C', 'B', 'B', 'B', 'C', 'C', 'C', 'C', 'C', 'C', 'B', 'B', 'A', 'A', 'A', 'A', 'B', 'C', 'B', 'B', 'B', 'A', 'A', 'C', 'C', 'C', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'C', 'B', 'C', 'B', 'C', 'C', 'C', 'C', 'C', 'B', 'B', 'C', 'B', 'A', 'A', 'B', 'A', 'C', 'C', 'B', 'B', 'B', 'A', 'A', 'A', 'A', 'B', 'C', 'A', 'C', 'C', 'C', 'B', 'C', 'B', 'C', 'B', 'B', 'B', 'B', 'C', 'B', 'B', 'B', 'C', 'C', 'C', 'B', 'B', 'A', 'A', 'A', 'A', 'B', 'A', 'C', 'C', 'B', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'B', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'B', 'C', 'B', 'B', 'C', 'C', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'C', 'B', 'B', 'A', 'A', 'B', 'A', 'A', 'A', 'A', 'C', 'A', 'B', 'B', 'C', 'B', 'A', 'A', 'A', 'A', 'A', 'A', 'C', 'B', 'C', 'C', 'C', 'C', 'C', 'C', 'A', 'A', 'C', 'A', 'B', 'B', 'B', 'A', 'C', 'B', 'A', 'B', 'C', 'A', 'A', 'B', 'B', 'A', 'B', 'A', 'A', 'C', 'C', 'B', 'B', 'C', 'C', 'A', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'A', 'A', 'A', 'B', 'B', 'B', 'A', 'C', 'C', 'B', 'A', 'A', 'A', 'B', 'A', 'B', 'B', 'B', 'B', 'B', 'B', 'C', 'C', 'B', 'A', 'A', 'B', 'C', 'B', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'C', 'C', 'C', 'C', 'C', 'B', 'A', 'B', 'A', 'A', 'A', 'A', 'B', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'A', 'A', 'A', 'A', 'A', 'A', 'C', 'C', 'B', 'C', 'C', 'B', 'B', 'B', 'C', 'C', 'B', 'A', 'A', 'A', 'C', 'B', 'A', 'B', 'B', 'B', 'B', 'A', 'A', 'B', 'A', 'A', 'A', 'B', 'B', 'B', 'A', 'A', 'A', 'A', 'A', 'A', 'B', 'C', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'A', 'A', 'A', 'B', 'B', 'C', 'B', 'B', 'A', 'A', 'A', 'C', 'A', 'A', 'A', 'C', 'B', 'A', 'B', 'C', 'C', 'A', 'C', 'B', 'C', 'B', 'A', 'A', 'A', 'B', 'A', 'B', 'C', 'C', 'B', 'B', 'B', 'B', 'C', 'C', 'C', 'C', 'B', 'B', 'B', 'C', 'C', 'B', 'A', 'A', 'A', 'A', 'B', 'A', 'A', 'C', 'B', 'B', 'C', 'C', 'C', 'B', 'A', 'A', 'B', 'A', 'C', 'B', 'B', 'B', 'B', 'A', 'B', 'C', 'B', 'B', 'C', 'B', 'B', 'C', 'B', 'B', 'A', 'A', 'C', 'B', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'C', 'B', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'A', 'A', 'B', 'A', 'A', 'B', 'C', 'C', 'C', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'C', 'C', 'B', 'B', 'B', 'A', 'A', 'A', 'A', 'A', 'C', 'B', 'A']
P, s = get_transition_matrix(my_sequence)

print("狀態空間：", s)
print("\n轉移矩陣 P：\n", P)