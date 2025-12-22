import torch
import torch.nn as nn
from torch.nn import functional as F
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import json
import random

# --- 超參數設定 ---
batch_size = 64
max_iters = 5000
learning_rate = 5e-4
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
n_embd = 128
n_head = 8
n_layer = 4
dropout = 0.1

# 影像規格
img_w, img_h = 64, 16
patch_size = 8
num_patches = (img_w // patch_size) * (img_h // patch_size) # 16 patches
patch_dim = patch_size * patch_size # 64
text_len = 10 # 指令最大長度，例如 "畫出 9836  "

# 詞彙表：支援中文指令與數字
chars = " 0123456789畫出寫"
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
vocab_size = len(chars)

os.makedirs('img_progress', exist_ok=True)

# 1. 數據生成模組
def get_batch(batch_size):
    texts, images = [], []
    for _ in range(batch_size):
        num = str(random.randint(0, 9999))
        verb = random.choice(["畫出", "寫出"])
        full_text = f"{verb} {num}".ljust(text_len)
        
        # 繪製對應數字的影像
        img = Image.new('L', (img_w, img_h), color=0)
        d = ImageDraw.Draw(img)
        d.text((2, 0), num, fill=255) # 只畫出數字部分
        
        texts.append(full_text)
        images.append(np.array(img).astype(np.float32) / 255.0)
        
    x_text = torch.tensor([[stoi[c] for c in t] for t in texts]).to(device)
    y_imgs = torch.tensor(np.array(images)).to(device)
    return x_text, y_imgs

def img_to_patches(img):
    B, H, W = img.shape
    p = patch_size
    patches = img.view(B, H//p, p, W//p, p).permute(0, 1, 3, 2, 4).contiguous()
    return patches.view(B, -1, p*p)

def patches_to_img(patches):
    B = patches.size(0)
    p = patch_size
    img = patches.view(B, 2, 8, p, p).permute(0, 1, 3, 2, 4).contiguous()
    return img.view(B, img_h, img_w)

# 2. 模型組件 (結合 CharGPT 與 ViT)
class T2ITransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Parameter(torch.zeros(1, text_len + num_patches, n_embd))
        self.dropout = nn.Dropout(dropout)
        
        # Transformer Blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_embd, nhead=n_head, batch_first=True, activation='gelu'
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
        
        self.ln_f = nn.LayerNorm(n_embd)
        self.patch_head = nn.Linear(n_embd, patch_dim)

    def forward(self, idx):
        B, T = idx.shape
        
        # 文字嵌入
        tok_emb = self.token_embedding(idx) # (B, T, n_embd)
        # 影像 Query (全零，模型會根據文字填入內容)
        img_query = torch.zeros(B, num_patches, n_embd, device=device)
        
        x = torch.cat((tok_emb, img_query), dim=1) # 合併序列
        x = x + self.position_embedding
        x = self.dropout(x)
        x = self.blocks(x)
        x = self.ln_f(x)
        
        # 只取後半段影像部分進行預測
        img_out = self.patch_head(x[:, T:, :])
        return torch.sigmoid(img_out)

# 3. 訓練流程
def train():
    model = T2ITransformer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    print(f"啟動訓練。設備: {device} | 參數量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    for iter in range(max_iters + 1):
        x_batch, y_batch = get_batch(batch_size)
        target_patches = img_to_patches(y_batch)
        
        preds = model(x_batch)
        loss = F.binary_cross_entropy(preds, target_patches)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iter % 500 == 0:
            print(f"迭代 {iter:4d}: Loss {loss.item():.4f}")
            # 存檔範例
            with torch.no_grad():
                test_idx = x_batch[:1] # 取 batch 中第一個
                sample_pred = model(test_idx)
                sample_img = patches_to_img(sample_pred).cpu().numpy()[0]
                res = Image.fromarray((sample_img * 255).astype(np.uint8))
                res.resize((img_w*4, img_h*4), resample=Image.NEAREST).save(f"img_progress/step_{iter}.png")

    # 儲存模型
    torch.save(model.state_dict(), "t2i_gpt_model.pth")
    # 儲存設定
    with open("t2i_config.json", "w") as f:
        json.dump({"n_embd": n_embd, "n_head": n_head, "n_layer": n_layer, "chars": chars}, f)
    print("模型訓練完成並已存檔。")

if __name__ == "__main__":
    train()