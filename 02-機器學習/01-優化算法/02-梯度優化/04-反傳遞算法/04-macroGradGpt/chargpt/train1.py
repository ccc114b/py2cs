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

    block_size = 16  # 稍微增加 context 長度
    batch_size = 8   # 稍微增加 batch size 提高梯度穩定性

    def get_batch():
        ix = np.random.randint(0, len(data) - block_size, (batch_size,))
        x = np.stack([data[i:i+block_size] for i in ix])
        y = np.stack([data[i+1:i+block_size+1] for i in ix])
        return x, y

    # 4. 初始化模型
    # 這裡的參數可以根據 CPU 速度調整，如果太慢可以減少 n_layer 或 n_embd
    model = GPT(vocab_size, n_embd=64, n_head=4, n_layer=2, block_size=block_size)

    def cross_entropy_loss(logits, targets):
        # logits: (B, T, V), targets: (B, T)
        B, T, V = logits.shape
        logits_flat = logits.data.reshape(B*T, V)
        targets_flat = targets.reshape(-1)
        
        # 數值穩定的 Softmax
        logits_max = np.max(logits_flat, axis=-1, keepdims=True)
        exps = np.exp(logits_flat - logits_max)
        probs = exps / np.sum(exps, axis=-1, keepdims=True)
        
        # 計算交叉熵 Loss
        # p = probs[每個樣本的正確 index]
        p = probs[range(B*T), targets_flat]
        loss_val = -np.mean(np.log(p + 1e-9))
        
        # 建立回傳用的 Tensor，連接到 logits
        out = Tensor(loss_val, _children=(logits,), _op='cross_entropy')

        def _backward():
            # 交叉熵的梯度公式: probs - y_onehot
            d_logits = probs.copy()
            d_logits[range(B*T), targets_flat] -= 1
            d_logits /= (B*T) # 平均值梯度
            logits.grad += d_logits.reshape(B, T, V)
        
        out._backward = _backward
        return out

    # 5. 訓練迴圈
    lr = 0.01   # 提高學習率
    # steps = 1000 # 建議至少 1000 步
    steps = 10000 # 建議至少 1000 步
    
    print(f"Starting training (steps={steps}, lr={lr})...")

    for step in range(steps):
        xb, yb = get_batch()
        
        # 前向傳播
        logits = model(xb)
        loss = cross_entropy_loss(logits, yb)
        
        # 反向傳播
        model.zero_grad()
        loss.backward()
        
        # 梯度裁剪 (防止 Transformer 梯度爆炸)
        grad_norm = 0
        params = model.parameters()
        for p in params:
            grad_norm += np.sum(p.grad**2)
        grad_norm = np.sqrt(grad_norm)
        
        if grad_norm > 1.0:
            for p in params:
                p.grad /= (grad_norm / 1.0)
        
        # 權重更新 (SGD)
        for p in params:
            p.data -= lr * p.grad
        
        if step % 50 == 0 or step == steps - 1:
            print(f"Step {step:4d} | Loss: {loss.data:.4f} | GradNorm: {grad_norm:.4f}")

    # 6. 儲存模型權重
    with open('gpt_weights.pkl', 'wb') as f:
        weights = [p.data for p in model.parameters()]
        pickle.dump(weights, f)
    print("Training finished. Weights and Meta saved.")

if __name__ == "__main__":
    main()