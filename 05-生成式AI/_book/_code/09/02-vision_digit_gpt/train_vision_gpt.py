import torch
import torch.nn as nn
from torch.nn import functional as F
from PIL import Image, ImageDraw
import numpy as np
import os
import json

# --- 1. 超參數設定 ---
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
n_embd, n_head, n_layer = 128, 8, 4
img_w, img_h = 64, 16
patch_size = 8
num_patches = (img_w // patch_size) * (img_h // patch_size)
patch_dim = patch_size * patch_size
text_len = 4
max_iters = 2000
batch_size = 64

# 建立必要的資料夾
os.makedirs('training_samples', exist_ok=True)

# 詞彙表
chars = "0123456789 "
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

# --- 2. 輔助函數 ---
def generate_batch(batch_size):
    texts, images = [], []
    for _ in range(batch_size):
        num = "".join([str(np.random.randint(0, 10)) for _ in range(np.random.randint(1, 5))])
        label_text = num.ljust(text_len)
        img = Image.new('L', (img_w, img_h), color=0)
        d = ImageDraw.Draw(img)
        d.text((2, 0), label_text.strip(), fill=255)
        images.append(np.array(img).astype(np.float32) / 255.0)
        texts.append(label_text)
    return texts, torch.tensor(np.array(images))

def img_to_patches(img):
    B, H, W = img.shape
    p = patch_size
    patches = img.view(B, H//p, p, W//p, p).permute(0, 1, 3, 2, 4).contiguous()
    return patches.view(B, -1, p*p)

def patches_to_img(patches):
    B, _, _ = patches.shape
    p = patch_size
    grid_w, grid_h = img_w // p, img_h // p
    img = patches.view(B, grid_h, grid_w, p, p).permute(0, 1, 3, 2, 4).contiguous()
    return img.view(B, img_h, img_w)

# --- 3. 模型定義 ---
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

# --- 4. 訓練循環 ---
model = T2ITransformer(len(chars)).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

print(f"開始訓練。設備: {device}")

for i in range(max_iters + 1):
    texts, imgs = generate_batch(batch_size)
    text_in = torch.tensor([[stoi.get(c, stoi[' ']) for c in t] for t in texts]).to(device)
    target_patches = img_to_patches(imgs.to(device))
    
    # Forward & Backward
    pred_patches = model(text_in)
    loss = F.binary_cross_entropy(pred_patches, target_patches)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 定期輸出範例
    if i % 200 == 0:
        print(f"迭代 {i:4d} | Loss: {loss.item():.4f}")
        
        with torch.no_grad():
            # 隨機挑選 3 組展示
            test_indices = [0, 1, 2]
            sample_texts = [texts[idx] for idx in test_indices]
            
            # 模型生成的預測圖
            pred_imgs = patches_to_img(pred_patches[test_indices]).cpu().numpy()
            # 原始真實圖
            real_imgs = imgs[test_indices].numpy()
            
            # 拼接一張對照圖 (左邊是真實，右邊是預測)
            canvas = Image.new('L', (img_w * 2 + 10, img_h * 3 + 10), color=128)
            for idx in range(3):
                r_img = Image.fromarray((real_imgs[idx] * 255).astype(np.uint8))
                p_img = Image.fromarray((pred_imgs[idx] * 255).astype(np.uint8))
                
                y_off = idx * (img_h + 2) + 2
                canvas.paste(r_img, (2, y_off))
                canvas.paste(p_img, (img_w + 5, y_off))
            
            # 放大 4 倍存檔，看得更清楚
            canvas.resize((canvas.width * 4, canvas.height * 4), resample=Image.NEAREST).save(f"training_samples/step_{i}.png")

# 儲存最終模型
torch.save(model.state_dict(), "num_t2i_model.pth")
with open("model_config.json", "w") as f:
    json.dump({"chars": chars, "n_embd": n_embd, "n_head": n_head, "n_layer": n_layer, "img_w": img_w, "img_h": img_h, "patch_size": patch_size, "text_len": text_len}, f)

print("訓練完成！模型已儲存。")