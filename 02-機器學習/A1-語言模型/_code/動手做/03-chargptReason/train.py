import torch
import json
import os
from model import CharGPT, get_device

# --- 針對 8GB Mac 優化的超參數 ---
batch_size = 32      # 若卡頓可調至 16
max_iters = 200     
eval_interval = 10  
learning_rate = 1e-3 # 提高學習率以加快收斂
eval_iters = 100
device = get_device()

# 模型規模參數
n_embd = 128         # 降低維度以換取速度
n_head = 4
n_layer = 4
block_size = 256
dropout = 0.1
# ------------------------------

# 1. 載入數據與編碼
print("正在載入數據集...")
with open('math_dataset.jsonl', 'r', encoding='utf-8') as f:
    raw_data = [json.loads(line)['text'] for line in f]

full_text = "\n".join(raw_data)
chars = sorted(list(set(full_text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]

# 轉換數據為 Tensor 並搬移到裝置 (預載到 MPS/CUDA 加速訓練)
data_tensor = torch.tensor(encode(full_text), dtype=torch.long)
n = int(0.9 * len(data_tensor))
train_data = data_tensor[:n]
val_data = data_tensor[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# 2. 初始化 CharGPT 模型
model = CharGPT(vocab_size, n_embd, n_head, n_layer, block_size, dropout, device)
model.to(device)
print(f"模型參數: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# 3. 訓練循環
print("開始訓練 (按 Ctrl+C 可隨時停止並儲存)...")
try:
    for iter in range(max_iters):
        if iter % eval_interval == 0:
            losses = estimate_loss(model)
            print(f"步數 {iter}: 訓練損失 {losses['train']:.4f}, 驗證損失 {losses['val']:.4f}")
            
            # 儲存包含超參數的完整 Checkpoint
            checkpoint = {
                'model_state': model.state_dict(),
                'config': {'vocab_size': vocab_size, 'n_embd': n_embd, 'n_head': n_head, 
                           'n_layer': n_layer, 'block_size': block_size, 'dropout': dropout},
                'stoi': stoi,
                'itos': itos
            }
            torch.save(checkpoint, 'chargpt_math.pth')

        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
except KeyboardInterrupt:
    print("\n停止訓練，正在儲存...")

print("完成！模型已儲存為 chargpt_math.pth")