import numpy as np

# 1. 定義模型參數
states = ['晴天', '陰天', '雨天']
state_to_index = {state: i for i, state in enumerate(states)}

initial_distribution = np.array([0.6, 0.3, 0.1])

transition_matrix = np.array([
    [0.7, 0.2, 0.1],  # 從晴天轉移
    [0.3, 0.5, 0.2],  # 從陰天轉移
    [0.1, 0.4, 0.5]   # 從雨天轉移
])

def generate_sequence(length):
    """
    根據馬可夫鏈模型產生一個指定長度的序列。
    """
    # 2. 選擇初始狀態
    current_state_index = np.random.choice(len(states), p=initial_distribution)
    sequence = [states[current_state_index]]

    # 3. 迭代生成後續狀態
    for _ in range(length - 1):
        # 根據當前狀態的轉移機率選擇下一個狀態
        next_state_index = np.random.choice(
            len(states),
            p=transition_matrix[current_state_index]
        )
        current_state_index = next_state_index
        sequence.append(states[current_state_index])

    return sequence

# 產生一個長度為 10 的序列
my_sequence = generate_sequence(10)
print("產生的序列：", my_sequence)