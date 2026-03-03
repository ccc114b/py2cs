import numpy as np
import sys
import pickle
import os

# 確保能找到專案目錄下的模組
sys.path.append(os.getcwd())

from macrograd.nn import GPT
from macrograd.engine import Tensor

def main():
    # 1. 檢查必要的檔案
    meta_path = 'meta.pkl'
    weights_path = 'gpt_weights.pkl'

    if not os.path.exists(meta_path) or not os.path.exists(weights_path):
        print("錯誤: 找不到 meta.pkl 或 gpt_weights.pkl，請先執行 train.py。")
        sys.exit(1)

    # 2. 載入詞彙表資訊 (Metadata)
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    
    char_to_idx = meta['char_to_idx']
    idx_to_char = meta['idx_to_char']
    vocab_size = meta['vocab_size']
    
    # 3. 初始化模型
    # 注意：這裡的參數 (n_embd, n_head, n_layer, block_size) 必須與 train.py 完全一致
    block_size = 16 
    model = GPT(vocab_size, n_embd=64, n_head=4, n_layer=2, block_size=block_size)

    # 4. 載入權重
    with open(weights_path, 'rb') as f:
        weights = pickle.load(f)
    
    for p, w_data in zip(model.parameters(), weights):
        if p.data.shape != w_data.shape:
            print(f"警告: 權重形狀不匹配! 模型:{p.data.shape}, 存檔:{w_data.shape}")
        p.data = w_data

    print("模型與權重載入完成。")

    # 5. 設定生成參數
    prompt = sys.argv[1] if len(sys.argv) > 1 else "問：" # 預設開頭
    max_new_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    temperature = 1.0 # 溫度越高越隨機，越低越保守

    # 6. 開始生成
    print(f"--- 生成開始 (起始字串: '{prompt}') ---")
    
    # 將起始字串編碼
    # 如果字元不在詞彙表中，則跳過
    idx_list = [char_to_idx[c] for c in prompt if c in char_to_idx]
    if not idx_list:
        idx_list = [0] # 如果 prompt 為空或無效，隨機從一個字開始
    
    x = np.array(idx_list)[None, :] # 建立 batch 維度 (1, T)

    result = list(prompt)
    
    for _ in range(max_new_tokens):
        # 如果當前長度超過 block_size，需要裁剪
        x_cond = x if x.shape[1] <= block_size else x[:, -block_size:]
        
        # 前向傳播取得 Logits
        logits = model(x_cond) # (1, T, V)
        
        # 只取最後一個時間步的預測值
        last_logits = logits.data[0, -1, :] / temperature
        
        # Softmax 轉機率
        exps = np.exp(last_logits - np.max(last_logits))
        probs = exps / np.sum(exps)
        
        # 根據機率分佈抽樣下一個字
        next_idx = np.random.choice(len(probs), p=probs)
        
        # 更新輸入序列並解碼
        x = np.concatenate([x, [[next_idx]]], axis=1)
        result.append(idx_to_char[next_idx])
        
        # 即時印出（可選）
        print(idx_to_char[next_idx], end='', flush=True)

    print("\n--- 生成結束 ---")

if __name__ == "__main__":
    main()