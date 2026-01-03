import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(SimpleLanguageModel, self).__init__()
        # 詞嵌入層 (Embedding Layer)：將離散的 Token 索引轉換為連續向量
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # 循環神經網路 (RNN)：處理序列資訊，保留上下文狀態
        self.rnn = nn.RNN(embed_size, hidden_size, batch_first=True)
        # 線性輸出層：將隱藏層狀態映射回詞彙表大小的 Logits
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h=None):
        # x: (Batch_size, Sequence_Length)
        x = self.embedding(x)  # (Batch_size, Seq_Len, Embed_Dim)
        out, h = self.rnn(x, h) # out: (Batch_size, Seq_Len, Hidden_Dim)
        logits = self.fc(out)   # (Batch_size, Seq_Len, Vocab_Size)
        return logits, h

def generate_text(model, start_token, max_len, temp=1.0, top_p=0.9):
    model.eval()
    input_seq = torch.tensor([[start_token]]) # 初始輸入
    generated = [start_token]
    hidden = None

    with torch.no_grad():
        for _ in range(max_len):
            # 1. 模型預測
            logits, hidden = model(input_seq, hidden)
            
            # 取得最後一個時間步的機率分佈，並加入溫度 (Temperature) 調整
            next_token_logits = logits[:, -1, :] / temp
            probs = F.softmax(next_token_logits, dim=-1)

            # 2. 核採樣 (Nucleus Sampling / Top-p Sampling)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            # 移除累積機率超過 p 的標記
            sorted_indices_to_remove = cumulative_probs > top_p
            # 保留第一個超過 p 的標記 (確保至少有一個可選)
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # 將不符合條件的機率設為 0 並重新歸一化
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            probs[0, indices_to_remove] = 0
            probs = probs / probs.sum()

            # 3. 隨機採樣 (Random Sampling)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated.append(next_token.item())
            input_seq = next_token # 自回歸 (Autoregressive)：將輸出作為下一步輸入

    return generated

# --- 模擬執行 ---
vocab_size = 50
model = SimpleLanguageModel(vocab_size=vocab_size, embed_size=16, hidden_size=32)
start_node = 1  # 假設 1 是開始符號 <SOS>
generated_sequence = generate_text(model, start_node, max_len=10)

print(f"生成的 Token 序列: {generated_sequence}")