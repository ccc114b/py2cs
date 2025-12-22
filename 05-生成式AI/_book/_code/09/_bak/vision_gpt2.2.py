import torch
import torch.nn as nn
from torch.nn import functional as F
from PIL import Image, ImageDraw
import numpy as np
import os

# 確保輸出目錄存在
if not os.path.exists('img'):
    os.makedirs('img')

# --- 1. 超參數設定 (針對 1 巷數字優化) ---
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
n_embd = 128          # 增加維度到 128，訓練會更穩定
n_head = 8
n_layer = 4
max_iters = 3000      
batch_size = 64       # 增加 Batch Size 穩定梯度
img_w, img_h = 64, 16 # 大幅縮小：寬 64, 高 16
patch_size = 8        # 8x8 patch
num_patches = (img_w // patch_size) * (img_h // patch_size) # 8 * 2 = 16 patches
patch_dim = patch_size * patch_size 
text_len = 4          # 只處理 4 個字

# 2. 數據生成器 (16x64 灰階圖)
def generate_batch(batch_size):
    texts = []
    images = []
    for _ in range(batch_size):
        # 隨機產生 1~4 位數
        num = "".join([str(np.random.randint(0, 10)) for _ in range(np.random.randint(1, 5))])
        label_text = num.ljust(text_len) # 用空格補齊到 4 位
        
        img = Image.new('L', (img_w, img_h), color=0)
        d = ImageDraw.Draw(img)
        # 繪製文字，座標 (2, 0) 讓它垂直置中
        d.text((2, 0), label_text.strip(), fill=255)
        
        img_array = np.array(img).astype(np.float32) / 255.0
        texts.append(label_text)
        images.append(img_array)
    return texts, torch.tensor(np.array(images))

def img_to_patches(img):
    B, H, W = img.shape
    p = patch_size
    # 重新排列成 patch 序列: (B, H/p, W/p, p, p) -> (B, num_patches, patch_dim)
    patches = img.view(B, H//p, p, W//p, p).permute(0, 1, 3, 2, 4).contiguous()
    return patches.view(B, -1, p*p)

def patches_to_img(patches):
    B, _, _ = patches.shape
    p = patch_size
    grid_w = img_w // p
    grid_h = img_h // p
    img = patches.view(B, grid_h, grid_w, p, p).permute(0, 1, 3, 2, 4).contiguous()
    return img.view(B, img_h, img_w)

# 3. Model
class T2ITransformer(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, text_len + num_patches, n_embd))
        
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=n_embd, nhead=n_head, dim_feedforward=n_embd*4, 
            batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=n_layer)
        self.out_head = nn.Linear(n_embd, patch_dim)

    def forward(self, text_idx):
        B = text_idx.size(0)
        t_emb = self.token_emb(text_idx)
        p_query = torch.zeros(B, num_patches, n_embd, device=device)
        
        x = torch.cat((t_emb, p_query), dim=1)
        x = x + self.pos_emb[:, :x.size(1), :]
        x = self.transformer(x)
        
        img_logits = self.out_head(x[:, text_len:, :])
        return torch.sigmoid(img_logits)

# 4. 訓練邏輯
chars = "0123456789 " # 包含空格作為 padding
stoi = { ch:i for i,ch in enumerate(chars) }
model = T2ITransformer(len(chars)).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

print(f"啟動模型: {device} | 影像大小: {img_w}x{img_h} | Patches: {num_patches}")

for i in range(max_iters + 1):
    texts, imgs = generate_batch(batch_size)
    # 這裡 stoi[' '] 用於處理找不到的字元
    text_in = torch.tensor([[stoi.get(c, stoi[' ']) for c in t] for t in texts]).to(device)
    target_patches = img_to_patches(imgs.to(device))
    
    pred_patches = model(text_in)
    loss = F.binary_cross_entropy(pred_patches, target_patches)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if i % 200 == 0:
        print(f"迭代 {i}/{max_iters}, Loss: {loss.item():.4f}")
        
        with torch.no_grad():
            # 測試用固定的數字
            sample_text = "1234".ljust(text_len)
            sample_idx = torch.tensor([[stoi.get(c, stoi[' ']) for c in sample_text]]).to(device)
            res_patches = model(sample_idx)
            res_img = patches_to_img(res_patches).cpu().numpy()[0]
            # 存檔時將圖片放大 4 倍方便觀看
            final_save = Image.fromarray((res_img * 255).astype(np.uint8))
            final_save.resize((img_w*4, img_h*4), resample=Image.NEAREST).save(f"img/progress_{i}.png")

# 5. 最終測試
test_str = "5678"
print(f"最終測試生成: {test_str}")
test_idx = torch.tensor([[stoi.get(c, stoi[' ']) for c in test_str.ljust(text_len)]]).to(device)
with torch.no_grad():
    final_patches = model(test_idx)
    final_img = patches_to_img(final_patches).cpu().numpy()[0]
    final_save = Image.fromarray((final_img * 255).astype(np.uint8))
    final_save.resize((img_w*4, img_h*4), resample=Image.NEAREST).save("img/final_output.png")