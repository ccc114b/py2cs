import torch
import os
import sys
import argparse
from model import CharGPT, get_device

# --- 超參數 ---
batch_size = 64
block_size = 128
max_iters = 1000  # 增加迭代次數
learning_rate = 3e-4
device = get_device()
n_embd = 192
n_head = 6
n_layer = 6
dropout = 0.2
SAVE_PATH = "gpt_model.pt"

def train_model(file_path):
    if not os.path.exists(file_path):
        print(f"錯誤：找不到檔案 '{file_path}'")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    encode = lambda s: [stoi[c] for c in s if c in stoi]

    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    def get_batch(split):
        d = train_data if split == 'train' else val_data
        ix = torch.randint(len(d) - block_size, (batch_size,))
        x = torch.stack([d[i:i+block_size] for i in ix])
        y = torch.stack([d[i+1:i+block_size+1] for i in ix])
        return x.to(device), y.to(device)

    model = CharGPT(vocab_size, n_embd, n_head, n_layer, block_size, dropout, device).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    print(f"開始訓練... 設備: {device}")
    for iter in range(max_iters):
        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if iter % 100 == 0:
            print(f"迭代 {iter:4d}: 損失 {loss.item():.4f}")

    # --- 儲存模型與配置 ---
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'vocab_size': vocab_size,
        'itos': itos,
        'stoi': stoi,
        'config': {
            'n_embd': n_embd, 'n_head': n_head, 'n_layer': n_layer,
            'block_size': block_size, 'dropout': dropout
        }
    }
    torch.save(checkpoint, SAVE_PATH)
    print(f"模型已存檔至 {SAVE_PATH}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True)
    args = parser.parse_args()
    train_model(args.file)