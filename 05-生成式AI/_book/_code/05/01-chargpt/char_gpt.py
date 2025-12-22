import torch
import torch.nn as nn
from torch.nn import functional as F
import argparse
import sys
import os

# --- 超參數設定 ---
batch_size = 64       # 同時訓練的序列數量
block_size = 128      # 上下文長度 (Context Window)
max_iters = 500 # 2000      # 訓練迭代次數
learning_rate = 3e-4
# device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
n_embd = 192          # 嵌入維度
n_head = 6            # 多頭注意力的頭數
n_layer = 6           # Transformer Block 的層數
dropout = 0.2
# ----------------

def train_model(file_path):
    # 1. 讀取指定檔案的文字內容
    if not os.path.exists(file_path):
        print(f"錯誤：找不到檔案 '{file_path}'")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    if len(text) < block_size + 2:
        print("錯誤：檔案內容太短，不足以進行訓練。")
        return

    print(f"數據加載成功，總字元數: {len(text)}")

    # 2. 建立詞彙表 (以字元為單位)
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    encode = lambda s: [stoi[c] for c in s if c in stoi]
    decode = lambda l: ''.join([itos[i] for i in l])

    # 轉換數據為 Tensor
    data = torch.tensor(encode(text), dtype=torch.long)
    
    # 劃分訓練與驗證集 (90% 訓練, 10% 驗證)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    def get_batch(split):
        d = train_data if split == 'train' else val_data
        ix = torch.randint(len(d) - block_size, (batch_size,))
        x = torch.stack([d[i:i+block_size] for i in ix])
        y = torch.stack([d[i+1:i+block_size+1] for i in ix])
        return x.to(device), y.to(device)

    # 3. 模型組件定義 (Causal GPT 結構)
    class CausalSelfAttention(nn.Module):
        def __init__(self, n_embd, n_head):
            super().__init__()
            self.key = nn.Linear(n_embd, n_embd, bias=False)
            self.query = nn.Linear(n_embd, n_embd, bias=False)
            self.value = nn.Linear(n_embd, n_embd, bias=False)
            self.proj = nn.Linear(n_embd, n_embd)
            self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size))
                                         .view(1, 1, block_size, block_size))
            self.n_head = n_head

        def forward(self, x):
            B, T, C = x.shape
            k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

            att = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1)**0.5))
            att = att.masked_fill(self.tril[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            y = att @ v
            y = y.transpose(1, 2).contiguous().view(B, T, C)
            return self.proj(y)

    class Block(nn.Module):
        def __init__(self, n_embd, n_head):
            super().__init__()
            self.sa = CausalSelfAttention(n_embd, n_head)
            self.ffn = nn.Sequential(
                nn.Linear(n_embd, 4 * n_embd),
                nn.ReLU(),
                nn.Linear(4 * n_embd, n_embd),
                nn.Dropout(dropout),
            )
            self.ln1 = nn.LayerNorm(n_embd)
            self.ln2 = nn.LayerNorm(n_embd)

        def forward(self, x):
            x = x + self.sa(self.ln1(x))
            x = x + self.ffn(self.ln2(x))
            return x

    class CharGPT(nn.Module):
        def __init__(self, vocab_size):
            super().__init__()
            self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
            self.position_embedding_table = nn.Embedding(block_size, n_embd)
            self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
            self.ln_f = nn.LayerNorm(n_embd)
            self.lm_head = nn.Linear(n_embd, vocab_size)

        def forward(self, idx, targets=None):
            B, T = idx.shape
            tok_emb = self.token_embedding_table(idx)
            pos_emb = self.position_embedding_table(torch.arange(T, device=device))
            x = tok_emb + pos_emb
            x = self.blocks(x)
            x = self.ln_f(x)
            logits = self.lm_head(x)

            loss = None
            if targets is not None:
                B, T, C = logits.shape
                logits = logits.view(B*T, C)
                targets = targets.view(B*T)
                loss = F.cross_entropy(logits, targets)
            return logits, loss

        def generate(self, idx, max_new_tokens):
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -block_size:]
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, idx_next), dim=1)
            return idx

    # 4. 執行訓練
    model = CharGPT(vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    print(f"模型參數量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    print("開始訓練...")

    for iter in range(max_iters):
        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if iter % 100 == 0:
            val_xb, val_yb = get_batch('val')
            _, val_loss = model(val_xb, val_yb)
            print(f"迭代 {iter:4d}: 訓練損失 {loss.item():.4f}, 驗證損失 {val_loss.item():.4f}")

    # 5. 生成結果測試
    print("\n--- 訓練完成，生成樣本 ---")
    start_context = torch.zeros((1, 1), dtype=torch.long, device=device) # 從第一個字元開始
    print(decode(model.generate(start_context, max_new_tokens=200)[0].tolist()))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="訓練字元級自回歸 GPT 模型")
    parser.add_argument("--file", type=str, required=True, help="文字訓練檔案的路徑 (例如 input.txt)")
    
    # 處理沒傳參數的情況
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
        
    args = parser.parse_args()
    train_model(args.file)