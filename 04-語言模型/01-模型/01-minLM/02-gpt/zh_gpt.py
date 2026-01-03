import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# --- 0. 超參數設定 (Hyperparameters) ---
batch_size = 32        # 每次訓練抓幾筆資料
block_size = 64        # 上下文長度 (一次看多少個字來預測下一個)
max_iters = 2000       # 訓練總迭代次數
eval_interval = 200    # 每幾次評估一次 Loss
learning_rate = 3e-4   # 學習率
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu' # 有 GPU 就用 GPU
print('device=', device)
eval_iters = 200       # 評估時跑幾輪取平均
n_embd = 128           # 嵌入向量維度 (Hidden Size)
n_head = 4             # Attention 頭數
n_layer = 4            # Transformer Block 層數
dropout = 0.2          # Dropout 機率

print(f"使用裝置: {device}")

# --- 1. 準備資料與 Tokenizer ---

# 如果沒有 input.txt，建立一個範例檔案 (這裡用長恨歌+重複字串模擬數據)
if not os.path.exists('input.txt'):
    print("找不到 input.txt，正在生成範例資料...")
    sample_text = """
    漢皇重色思傾國，御宇多年求不得。楊家有女初長成，養在深閨人未識。
    天生麗質難自棄，一朝選在君王側。回眸一笑百媚生，六宮粉黛無顏色。
    春寒賜浴華清池，溫泉水滑洗凝脂。侍兒扶起嬌無力，始是新承恩澤時。
    雲鬢花顏金步搖，芙蓉帳暖度春宵。春宵苦短日高起，從此君王不早朝。
    承歡侍宴無閑暇，春從春遊夜專夜。後宮佳麗三千人，三千寵愛在一身。
    GPT模型原理是基於Transformer架構，透過注意力機制學習文字之間的關聯。
    深度學習是人工智慧的一個分支，它試圖模擬人腦的神經網絡。
    今天天氣真好，我們一起去公園散步吧。機器學習需要大量的數據。
    """ * 100 # 重複多次以增加數據量
    with open('input.txt', 'w', encoding='utf-8') as f:
        f.write(sample_text)

# 讀取檔案
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# 建立詞彙表 (Character-level Tokenizer)
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"資料總字數: {len(text)}, 詞彙表大小: {vocab_size}")

# 建立映射表
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s if c in stoi] # 字串 -> 數字列表
decode = lambda l: ''.join([itos[i] for i in l])     # 數字列表 -> 字串

# 分割訓練集與驗證集
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# 資料批次載入器
def get_batch(split):
    data = train_data if split == 'train' else val_data
    # 隨機選取 batch_size 個起始位置
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # 取出 x (輸入) 和 y (目標，即 x 向後位移一格)
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# --- 2. 定義模型組件 (Attention & Transformer Block) ---

class Head(nn.Module):
    """ 單個 Self-Attention Head """
    def __init__(self, head_size):
        super().__init__()
        # 這裡就是產生 Q, K, V 的線性層
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # 註冊一個下三角遮罩矩陣 (tril)，確保模型只能看見「過去」
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x input: (batch, time-step, channels/embd)
        B, T, C = x.shape
        
        # 1. 產生 Q, K, V
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        
        # 2. 計算 Attention Score (QK^T) 並縮放
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5 # (B, T, T)
        
        # 3. 遮罩 (Masking)：未來的 Token 不能被看見
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        
        # 4. Softmax 取得權重
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        # 5. 與 V 加權求和 (Weighted Aggregation)
        v = self.value(x) # (B, T, head_size)
        out = wei @ v     # (B, T, head_size)
        return out

class MultiHeadAttention(nn.Module):
    """ 多頭注意力機制：並行運作多個 Head """
    def __init__(self, num_heads, head_size):
        super().__init__()
        # 創建 num_heads 個獨立的 Head
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        # 投影層：將多個 Head 的結果拼接後，融合回 n_embd 維度
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 將每個 Head 的輸出在最後一維拼接 (Concat)
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """ 前饋神經網絡 (FFN) """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), # 擴展維度 (通常是 4 倍)
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), # 縮回原維度
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer Block: Communication (Attention) + Computation (FFN) """
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size) # 自注意力層
        self.ffwd = FeedForward(n_embd)                 # 前饋層
        self.ln1 = nn.LayerNorm(n_embd)                 # 正規化層 1
        self.ln2 = nn.LayerNorm(n_embd)                 # 正規化層 2

    def forward(self, x):
        # 殘差連接 (Residual Connection): x + ...
        # 注意：這裡是 Pre-norm 架構 (先 LayerNorm 再進層)
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# --- 3. GPT 主模型 ---

class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. 詞嵌入表 (Token Embedding)
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # 2. 位置編碼表 (Position Embedding)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # 3. 堆疊 Transformer Blocks
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        # 4. 最終的 LayerNorm
        self.ln_f = nn.LayerNorm(n_embd) 
        # 5. 輸出層 (Language Modeling Head) -> 映射回詞彙表大小
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # 初始化權重 (選擇性優化)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx, targets 都是 (B, T) 的整數張量
        tok_emb = self.token_embedding_table(idx) # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, n_embd)
        x = tok_emb + pos_emb # token 與 position 資訊相加
        
        # 通過 Transformer Blocks
        x = self.blocks(x) # (B, T, n_embd)
        x = self.ln_f(x)   # (B, T, n_embd)
        
        # 計算 logits
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            # 計算 Loss (Cross Entropy)
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx 是當前的 context (B, T)
        for _ in range(max_new_tokens):
            # 裁剪 context，確保不超過 block_size
            idx_cond = idx[:, -block_size:]
            
            # 取得預測 logits
            logits, _ = self(idx_cond)
            
            # 關注最後一個時間步 (預測下一個字)
            logits = logits[:, -1, :] # (B, C)
            
            # 應用 Softmax 轉為機率
            probs = F.softmax(logits, dim=-1) # (B, C)
            
            # 依機率取樣 (Sample)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            
            # 將新預測的字接在後面
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

# --- 4. 訓練流程 (Training) ---

model = GPTLanguageModel()
model = model.to(device)

# 建立優化器
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print(f"模型參數數量: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")
print("開始訓練...")

for iter in range(max_iters):
    
    # 定期評估 Loss
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # 取樣一個 Batch
    xb, yb = get_batch('train')

    # Forward + Backward
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# --- 5. 儲存模型 ---
torch.save(model.state_dict(), 'mini_gpt_chinese.pth')
print("訓練完成，模型已儲存為 'mini_gpt_chinese.pth'")

# --- 6. 接龍生成範例 (Inference) ---

print("\n--- GPT 自動接龍演示 ---")
context_str = "今天天氣" # 起始語句
print(f"輸入開頭: {context_str}")

# 將輸入字串轉為 tensor
context = torch.tensor([encode(context_str)], dtype=torch.long, device=device)

# 生成 100 個字
generated_ids = model.generate(context, max_new_tokens=100)
generated_text = decode(generated_ids[0].tolist())

print(f"GPT 接龍結果:\n{generated_text}")