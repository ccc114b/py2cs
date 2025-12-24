import torch
import torch.nn as nn
from torch.nn import functional as F
from PIL import Image, ImageDraw
import numpy as np
import os, json, random

# --- 參數設定 ---
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
n_embd, n_head, n_layer = 128, 8, 4
img_w, img_h, patch_size = 64, 16, 8
text_len = 15 # 指令最大長度
max_iters = 5000
batch_size = 64

# 1. 讀取檔案並建立動態字元集
if not os.path.exists("dataset.txt"):
    print("請先執行 generate_data.py")
    exit()

with open("dataset.txt", "r", encoding="utf-8") as f:
    lines = [line.strip().split('|') for line in f.readlines()]

# 收集所有出現過的字元
all_text = "".join([l[0] for l in lines]) + "0123456789 "
unique_chars = sorted(list(set(all_text)))
stoi = { ch:i for i,ch in enumerate(unique_chars) }
itos = { i:ch for i,ch in enumerate(unique_chars) }
vocab_size = len(unique_chars)

print(f"字元集建立完成，共 {vocab_size} 個字元。")

# 2. 數據批次處理
def get_batch():
    batch_lines = random.sample(lines, batch_size)
    x_texts, y_imgs = [], []
    for cmd, num in batch_lines:
        # 文字處理
        idx = [stoi.get(c, stoi[' ']) for c in cmd[:text_len].ljust(text_len)]
        x_texts.append(idx)
        
        # 影像處理 (根據指令中的數字繪圖)
        img = Image.new('L', (img_w, img_h), color=0)
        d = ImageDraw.Draw(img)
        d.text((2, 0), num, fill=255)
        y_imgs.append(np.array(img).astype(np.float32) / 255.0)
        
    return torch.tensor(x_texts).to(device), torch.tensor(np.array(y_imgs)).to(device)

# 3. 模型定義
class DynamicT2I(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, text_len + 16, n_embd))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=n_embd, nhead=n_head, batch_first=True, activation='gelu'),
            num_layers=n_layer
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.patch_head = nn.Linear(n_embd, 64)

    def forward(self, idx):
        B = idx.size(0)
        x = torch.cat((self.token_emb(idx), torch.zeros(B, 16, n_embd, device=device)), dim=1)
        x = self.transformer(x + self.pos_emb)
        return torch.sigmoid(self.patch_head(self.ln_f(x[:, text_len:, :])))

# 4. 訓練
model = DynamicT2I().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

os.makedirs('training_img', exist_ok=True)
print("開始訓練...")
for i in range(max_iters + 1):
    xb, yb = get_batch()
    # Patching
    B = yb.shape[0]
    target = yb.view(B, 2, 8, 8, 8).permute(0, 1, 3, 2, 4).reshape(B, 16, 64)
    
    loss = F.binary_cross_entropy(model(xb), target)
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    
    if i % 500 == 0:
        print(f"迭代 {i}, Loss: {loss.item():.4f}")
        # 儲存進度預覽
        with torch.no_grad():
            sample_p = model(xb[:1])
            sample_img = sample_p.view(1, 2, 8, 8, 8).permute(0, 1, 3, 2, 4).reshape(16, 64).cpu().numpy()
            Image.fromarray((sample_img * 255).astype(np.uint8)).save(f"training_img/iter_{i}.png")

# 5. 存檔 (包含字元集)
torch.save(model.state_dict(), "dynamic_t2i.pth")
with open("dynamic_config.json", "w", encoding="utf-8") as f:
    json.dump({
        "chars": unique_chars, "n_embd": n_embd, "n_head": n_head, "n_layer": n_layer,
        "text_len": text_len, "img_w": img_w, "img_h": img_h
    }, f, ensure_ascii=False)
print("模型與動態字元集已存檔。")