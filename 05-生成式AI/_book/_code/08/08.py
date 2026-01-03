import math
import random
from collections import Counter

# --- 8.2 思維鏈 (Chain-of-Thought) 與 自一致性 (Self-consistency) ---

class SimpleLLM:
    """模擬一個具備隨機性的語言模型，針對數學問題給出不同的思維鏈路徑"""
    def __init__(self):
        # 模擬針對問題 '15 + 27' 的不同推理路徑與結果
        self.responses = [
            {"cot": "10+20=30, 5+7=12, 30+12=42", "answer": 42},
            {"cot": "15+20=35, 35+7=42", "answer": 42},
            {"cot": "15+30=45, 45-3=42", "answer": 42},
            {"cot": "10+20=30, 5+7=13, 30+13=43", "answer": 43}, # 模擬錯誤路徑
        ]

    def generate(self, prompt):
        # 隨機採樣模擬 P(a|z, q)P(z|q) 的過程
        return random.choice(self.responses)

def self_consistency_demo(n_samples=5):
    llm = SimpleLLM()
    samples = []
    
    print(f"--- 自一致性 (Self-consistency) 採樣 (n={n_samples}) ---")
    for i in range(n_samples):
        res = llm.generate("15 + 27")
        samples.append(res['answer'])
        print(f"採樣 {i+1}: 推理 = {res['cot']} | 答案 = {res['answer']}")
    
    # 執行多數決 (Majority Voting): argmax_a \sum I(a_i = a)
    vote_counts = Counter(samples)
    final_answer = vote_counts.most_common(1)[0][0]
    
    print(f"\n投票結果: {dict(vote_counts)}")
    print(f"最終決定答案: {final_answer}\n")


# --- 8.4 蒙地卡羅樹搜尋 (Monte Carlo Tree Search, MCTS) ---

class MCTSNode:
    """MCTS 節點，儲存訪問次數 N 與 價值評估 V"""
    def __init__(self, state, parent=None):
        self.state = state          # 當前推理狀態
        self.parent = parent        # 父節點
        self.children = []          # 子節點 (可能的下一步推理)
        self.visit_count = 0        # N(s, a)
        self.total_value = 0.0      # 用於計算 V(s, a)

    @property
    def value(self):
        """計算平均價值 V(s, a)"""
        return self.total_value / self.visit_count if self.visit_count > 0 else 0

def uct_score(node, exploration_c=1.414):
    """計算 UCT 值: V + C * sqrt(ln(N_parent) / N_child)"""
    if node.visit_count == 0:
        return float('inf')  # 優先探索未訪問過的節點
    
    exploitation = node.value
    exploration = exploration_c * math.sqrt(math.log(node.parent.visit_count) / node.visit_count)
    return exploitation + exploration

class ReasoningMCTS:
    """模擬推理空間的 MCTS 搜尋"""
    def __init__(self, root_state):
        self.root = MCTSNode(root_state)

    def select(self, node):
        """選擇階段：根據 UCT 選擇子節點"""
        while node.children:
            node = max(node.children, key=uct_score)
        return node

    def expand(self, node):
        """擴展階段：生成可能的下一步 (範例中固定生成兩個後繼狀態)"""
        if node.visit_count > 0 or node == self.root:
            for i in range(2):
                new_state = f"{node.state} -> Step_{len(node.children)+1}"
                child = MCTSNode(new_state, parent=node)
                node.children.append(child)
        return random.choice(node.children) if node.children else node

    def simulate(self, node):
        """模擬階段：隨機評估該路徑的獎勵 (Reward)"""
        return random.uniform(0, 1)

    def backpropagate(self, node, reward):
        """回傳階段：更新路徑上的 N 與 V"""
        while node is not None:
            node.visit_count += 1
            node.total_value += reward
            node = node.parent

    def run(self, iterations=100):
        for _ in range(iterations):
            leaf = self.select(self.root)
            child = self.expand(leaf)
            reward = self.simulate(child)
            self.backpropagate(child, reward)

    def get_best_path(self):
        """獲取搜尋後訪問次數最多的路徑"""
        path = []
        curr = self.root
        while curr.children:
            curr = max(curr.children, key=lambda n: n.visit_count)
            path.append((curr.state, curr.visit_count))
        return path

# --- 執行程式 ---

if __name__ == "__main__":
    # 1. 示範 CoT 與自一致性
    self_consistency_demo(n_samples=10)

    # 2. 示範 MCTS 推理搜尋
    print("--- 蒙地卡羅樹搜尋 (MCTS) 推理路徑 ---")
    mcts = ReasoningMCTS("Start_Problem")
    mcts.run(iterations=200)
    
    best_path = mcts.get_best_path()
    for state, visits in best_path:
        print(f"狀態: {state} | 訪問次數: {visits}")