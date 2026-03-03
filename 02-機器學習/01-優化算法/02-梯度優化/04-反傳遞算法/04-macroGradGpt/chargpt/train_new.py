import numpy as np
import sys
import pickle
import os

# 確保能找到專案目錄下的模組
sys.path.append(os.getcwd())

from macrograd.nn import GPT
from macrograd.engine import Tensor

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m chargpt.train <corpus_file>")
        sys.exit(1)

    input_file = sys.argv[1]

    # 1. 讀取資料
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    # 2. 建立詞彙表 (Metadata)
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}

    print(f"Data loaded. File: {input_file}")
    print(f"Corpus length: {len(text)}, Vocab size: {vocab_size}")

    # 儲存 Meta 資訊供 predict.py 使用
    meta = {
        'chars': chars,
        'char_to_idx': char_to_idx,
        'idx_to_char': idx_to_char,
        'vocab_size': vocab_size
    }
    with open('meta.pkl', 'wb') as f:
        pickle.dump(meta, f)

    # 3. 資料編碼與批次準備
    def encode(s): return [char_to_idx[c] for c in s]
    data = np.array(encode(text))

    # --- 參數調整 ---
    block_size = 32  # 增加上下文長度至 32
    batch_size = 12  # 稍微加大 batch size
    n_embd = 128     # 增加維度
    n_head = 4
    n_layer = 3      # 增加層數
    # ---------------

    def get_batch():
        ix = np.random.randint(0, len(data) - block_size, (batch_size,))
        x = np.stack([data[i:i+block_size] for i in ix])
        y = np.stack([data[i+1:i+block_size+1] for i in ix])
        return x, y

    # 4. 初始化模型
    print(f"Initializing GPT (embd:{n_embd}, layer:{n_layer}, block:{block_size})...")
    model = GPT(vocab_size, n_embd=n_embd, n_head=n_head, n_layer=n_layer, block_size=block_size)

    def cross_entropy_loss(logits, targets):
        B, T, V = logits.shape
        logits_flat = logits.data.reshape(B*T, V)
        targets_flat = targets.reshape(-1)
        
        logits_max = np.max(logits_flat, axis=-1, keepdims=True)
        exps = np.exp(logits_flat - logits_max)
        probs = exps / np.sum(exps, axis=-1, keepdims=True)
        
        p = probs[range(B*T), targets_flat]
        loss_val = -np.mean(np.log(p + 1e-9))
        
        out = Tensor(loss_val, _children=(logits,), _op='cross_entropy')

        def _backward():
            d_logits = probs.copy()
            d_logits[range(B*T), targets_flat] -= 1
            d_logits /= (B*T)
            logits.grad += d_logits.reshape(B, T, V)
        
        out._backward = _backward
        return out

    # 5. 訓練迴圈設定
    lr = 0.01
    steps = 5000 # 2000 
    beta = 0.9  # Momentum 參數
    
    # 初始化 Momentum 的速度向量
    params = model.parameters()
    m = [np.zeros_like(p.data) for p in params]
    
    print(f"Starting training with SGD+Momentum (steps={steps}, lr={lr})...")
    print("Note: Larger model on CPU will be slower.")

    for step in range(steps):
        xb, yb = get_batch()
        
        # 前向傳播
        logits = model(xb)
        loss = cross_entropy_loss(logits, yb)
        
        # 反向傳播
        model.zero_grad()
        loss.backward()
        
        # 梯度裁剪 (Gradient Clipping)
        grad_norm = 0
        for p in params:
            grad_norm += np.sum(p.grad**2)
        grad_norm = np.sqrt(grad_norm)
        
        if grad_norm > 1.0:
            for p in params:
                p.grad /= (grad_norm / 1.0)
        
        # 6. 使用 SGD + Momentum 更新權重
        for i, p in enumerate(params):
            m[i] = beta * m[i] + (1 - beta) * p.grad  # 更新速度
            p.data -= lr * m[i]  # 更新權重
        
        if step % 50 == 0 or step == steps - 1:
            print(f"Step {step:4d} | Loss: {loss.data:.4f} | GradNorm: {grad_norm:.4f}")

    # 7. 儲存模型權重
    with open('gpt_weights.pkl', 'wb') as f:
        weights = [p.data for p in params]
        pickle.dump(weights, f)
    print("Training finished. Weights and Meta saved.")

if __name__ == "__main__":
    main()