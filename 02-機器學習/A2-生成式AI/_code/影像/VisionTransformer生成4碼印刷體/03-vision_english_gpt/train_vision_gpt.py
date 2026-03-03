import torch
import torch.nn as nn
from torch.nn import functional as F
from PIL import Image, ImageDraw
import numpy as np
import os
import json
import string
import random

# --- 1. 核心參數 (完全沿用您最滿意的數字模型設定) ---
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
n_embd = 128          
n_head = 8
n_layer = 4
img_w, img_h = 64, 16
patch_size = 8        
num_patches = 16 # (64//8) * (16//8) = 2 * 8
patch_dim = 64 # 8*8
text_len = 4          
max_iters = 10000     # 增加迭代，因為全字元集複雜度較高
batch_size = 64

os.makedirs('img', exist_ok=True)

# 擴大字元集：數字 + 大小寫英文 + 標點符號 + 空格
chars = string.digits + string.ascii_letters + string.punctuation + " "
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
vocab_size = len(chars)

# --- 2. 數據生成 ---
def generate_batch(batch_size):
    texts, images = [], []
    for _ in range(batch_size):
        curr_len = random.randint(1, 4)
        # 隨機組合全字元集
        rand_str = "".join(random.choice(chars[:-1]) for _ in range(curr_len))
        label_text = rand_str.ljust(text_len)
        
        img = Image.new('L', (img_w, img_h), color=0)
        d = ImageDraw.Draw(img)
        # 固定座標繪製，幫助快速收斂
        d.text((2, 0), label_text.strip(), fill=255)
        
        img_array = np.array(img).astype(np.float32) / 255.0
        texts.append(label_text)
        images.append(img_array)
    return texts, torch.tensor(np.array(images))

def img_to_patches(img):
    B, H, W = img.shape
    p = patch_size
    # (B, 16, 64) -> (B, 2, 8, 8, 8) -> (B, 16, 64)
    patches = img.view(B, H//p, p, W//p, p).permute(0, 1, 3, 2, 4).contiguous()
    return patches.view(B, -1, p*p)

def patches_to_img(patches):
    B, _, _ = patches.shape
    p = patch_size
    # 影像拼回 (B, 2, 8, 8, 8)
    img = patches.view(B, 2, 8, p, p).permute(0, 1, 3, 2, 4).contiguous()
    return img.view(B, img_h, img_w)

# --- 3. 模型結構 ---
class T2ITransformer(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, text_len + num_patches, n_embd))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=n_embd, nhead=n_head, batch_first=True, activation='gelu'),
            num_layers=n_layer
        )
        self.out_head = nn.Linear(n_embd, patch_dim)

    def forward(self, text_idx):
        B = text_idx.size(0)
        t_emb = self.token_emb(text_idx)
        p_query = torch.zeros(B, num_patches, n_embd, device=device)
        x = torch.cat((t_emb, p_query), dim=1) + self.pos_emb
        x = self.transformer(x)
        return torch.sigmoid(self.out_head(x[:, text_len:, :]))

# --- 4. 訓練邏輯 ---
model = T2ITransformer(vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

print(f"啟動全字元集訓練。設備: {device} | 總字元種數: {vocab_size}")

for i in range(max_iters + 1):
    texts, imgs = generate_batch(batch_size)
    text_in = torch.tensor([[stoi.get(c, stoi[' ']) for c in t] for t in texts]).to(device)
    target_patches = img_to_patches(imgs.to(device))
    
    pred_patches = model(text_in)
    loss = F.binary_cross_entropy(pred_patches, target_patches)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # 每 500 次迭代輸出 5 組對照範例
    if i % 500 == 0:
        print(f"迭代 {i:5d} | Loss: {loss.item():.4f}")
        with torch.no_grad():
            # 隨機挑選當前 batch 中的 5 個索引
            indices = [random.randint(0, batch_size - 1) for _ in range(5)]
            
            # 真實圖片 vs 預測圖片
            real_samples = imgs[indices].cpu().numpy()
            pred_samples = patches_to_img(pred_patches[indices]).cpu().numpy()
            
            # 建立一個大畫布：左邊放真實，右邊放預測 (中間留一點白線)
            # 每行寬度: 64*2 + 5 (間距) = 133, 高度: 16 * 5 + 間距 = 90
            canvas = Image.new('L', (img_w * 2 + 5, img_h * 5 + 10), color=128)
            for idx in range(5):
                r_img = Image.fromarray((real_samples[idx] * 255).astype(np.uint8))
                p_img = Image.fromarray((pred_samples[idx] * 255).astype(np.uint8))
                
                y_offset = idx * (img_h + 2) + 2
                canvas.paste(r_img, (0, y_offset))
                canvas.paste(p_img, (img_w + 5, y_offset))
            
            # 放大存檔方便觀看
            canvas.resize((canvas.width * 4, canvas.height * 4), resample=Image.NEAREST).save(f"img/step_{i}.png")

# 存檔
torch.save(model.state_dict(), "full_char_model.pth")
with open("full_char_config.json", "w") as f:
    json.dump({
        "chars": chars, "n_embd": n_embd, "n_head": n_head, "n_layer": n_layer, 
        "img_w": img_w, "img_h": img_h, "patch_size": patch_size, "text_len": text_len
    }, f)
print("模型與設定檔已儲存。")