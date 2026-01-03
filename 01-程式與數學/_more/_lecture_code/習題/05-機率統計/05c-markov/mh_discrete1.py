import numpy as np
import matplotlib.pyplot as plt

# --- 1. 設定目標分佈 (未歸一化) ---
# 目標分佈的相對機率。我們只知道這些值，不知道總和。
target_probs = {'A': 0.2, 'B': 0.5, 'C': 0.1, 'D': 0.2}
states = list(target_probs.keys())

def get_target_prob(state):
    return target_probs.get(state, 0)

# --- 2. 設定提議分佈 (Proposal Distribution) ---
# 這是一個簡單的對稱提議分佈：從當前狀態隨機跳到任何其他狀態
def proposal(current_state):
    # 建立一個包含除了當前狀態之外的所有狀態列表
    other_states = [s for s in states if s != current_state]
    # 從中隨機選擇一個作為提議狀態
    return np.random.choice(other_states)

# --- 3. 執行 Metropolis-Hastings 演算法 ---
def metropolis_hastings_discrete(n_samples, n_burnin):
    # 初始化一個隨機的起始狀態
    current_state = np.random.choice(states)
    
    samples = []
    
    for i in range(n_samples + n_burnin):
        # 產生一個候選狀態
        proposed_state = proposal(current_state)

        # 計算接受率
        # P(x') / P(x)
        acceptance_ratio = min(1, get_target_prob(proposed_state) / get_target_prob(current_state))
        
        # 決定是否接受
        if np.random.uniform(0, 1) < acceptance_ratio:
            current_state = proposed_state
        
        # 儲存樣本 (捨棄 burn-in 階段)
        if i >= n_burnin:
            samples.append(current_state)
            
    return samples

# --- 4. 執行並分析結果 ---
n_samples_val = 50000
n_burnin_val = 10000
samples = metropolis_hastings_discrete(n_samples_val, n_burnin_val)

# 計算每個狀態的頻率
from collections import Counter
sample_counts = Counter(samples)
total_samples = len(samples)
sample_probs = {state: count / total_samples for state, count in sample_counts.items()}

# 比較 M-H 樣本分佈與目標分佈
actual_probs = {state: prob / sum(target_probs.values()) for state, prob in target_probs.items()}

print("--- M-H 樣本分佈 ---")
for state in states:
    print(f"狀態 {state}: 樣本頻率 = {sample_probs.get(state, 0):.4f}, 理論機率 = {actual_probs[state]:.4f}")

# 可視化結果
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(8, 6))

x = np.arange(len(states))
width = 0.35

ax.bar(x - width/2, [sample_probs.get(s, 0) for s in states], width, label='M-H 樣本頻率', color='skyblue')
ax.bar(x + width/2, [actual_probs[s] for s in states], width, label='理論機率', color='coral')

ax.set_xticks(x)
ax.set_xticklabels(states)
ax.set_title('M-H 演算法在離散分佈上的應用')
ax.set_xlabel('狀態')
ax.set_ylabel('機率')
ax.legend()
plt.show()