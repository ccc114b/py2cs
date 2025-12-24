import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class SimpleLanguageModel(nn.Module):
    """
    這是一個極簡化的語言模型範例。
    在實際應用中，這會是一個 Transformer 模型。
    這裡我們用一個簡單的線性層來模擬對詞彙的 Logits 輸出。
    """
    def __init__(self, vocab_size, embed_dim):
        super(SimpleLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, input_ids):
        # 模擬模型計算過程
        x = self.embedding(input_ids)
        logits = self.lm_head(x)
        return logits

def get_log_prob(logits, labels):
    """
    計算給定標籤的對數機率 (Log probability)。
    """
    # 使用 LogSoftmax 獲取對數機率
    log_probs = F.log_softmax(logits, dim=-1)
    # 提取對應標籤的機率值
    per_token_logps = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return per_token_logps.sum(dim=-1)

def dpo_loss(policy_logps_w, policy_logps_l, ref_logps_w, ref_logps_l, beta=0.1):
    """
    實作 DPO 損失函數。
    """
    # 計算當前模型與參考模型的對數比值 (Log ratio)
    policy_log_ratios = policy_logps_w - policy_logps_l
    ref_log_ratios = ref_logps_w - ref_logps_l
    
    # 計算 DPO 特有的機率差值
    logits = policy_log_ratios - ref_log_ratios
    
    # 計算損失 (Loss)
    loss = -F.logsigmoid(beta * logits).mean()
    return loss

# --- 模擬訓練流程 ---

# 1. 參數設定
vocab_size = 100
embed_dim = 16
beta = 0.5

# 2. 初始化模型：一個是待訓練的策略模型 (Policy Model)，一個是凍結的參考模型 (Reference Model)
policy_model = SimpleLanguageModel(vocab_size, embed_dim)
reference_model = SimpleLanguageModel(vocab_size, embed_dim)
# 參考模型不更新參數
reference_model.eval()

optimizer = optim.Adam(policy_model.parameters(), lr=1e-3)

# 3. 模擬數據
# x: 問題的 Token ID
# y_w: 較好的回答 (Winning response)
# y_l: 較差的回答 (Losing response)
dummy_x = torch.randint(0, vocab_size, (2, 5)) 
dummy_y_w = torch.randint(0, vocab_size, (2, 8))
dummy_y_l = torch.randint(0, vocab_size, (2, 8))

# 4. 執行一次訓練迭代
optimizer.zero_grad()

# 計算當前模型的 Log Probs
policy_logits_w = policy_model(dummy_y_w)
policy_logits_l = policy_model(dummy_y_l)
policy_logps_w = get_log_prob(policy_logits_w, dummy_y_w)
policy_logps_l = get_log_prob(policy_logits_l, dummy_y_l)

# 計算參考模型的 Log Probs (不計算梯度)
with torch.no_grad():
    ref_logits_w = reference_model(dummy_y_w)
    ref_logits_l = reference_model(dummy_y_l)
    ref_logps_w = get_log_prob(ref_logits_w, dummy_y_w)
    ref_logps_l = get_log_prob(ref_logits_l, dummy_y_l)

# 5. 計算 DPO Loss 並更新
loss = dpo_loss(policy_logps_w, policy_logps_l, ref_logps_w, ref_logps_l, beta=beta)
loss.backward()
optimizer.step()

print(f"DPO 訓練迭代完成，當前 Loss: {loss.item():.4f}")