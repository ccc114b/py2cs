import torch
import torch.nn as nn
from torch.nn import functional as F
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import random

# --- 1. 超參數設定 ---
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
n_embd = 256
n_head = 8
n_layer = 6
max_iters = 5000      # 數字範圍小，5000 次應該能看到輪廓
batch_size = 32
img_size = 64
patch_size = 8        # 8x8 patch
num_patches = (img_size // patch_size) ** 2 
patch_dim = patch_size * patch_size 
text_len = 6          # 縮短長度，只練 6 位數以內的數字

# 只練數字與空格
all_chars = "0123456789 "
stoi = { ch:i for i,ch in enumerate(all_chars) }
itos = { i:ch for i,ch in enumerate(all_chars) }
vocab_size = len(all_chars)

# --- 2. 數據生成器 (專注於數字) ---
def generate_batch(batch_size):
    texts = []
    images = []
    for _ in range(batch_size):
        # 隨機產生 1-6 位的數字字串
        num_val = str(random.randint(0, 999999))
        label_text = num_val.ljust(text_len)
        
        img = Image.new('L', (img_size, img_size), color=0)
        d = ImageDraw.Draw(img)
        
        # 繪製文字：固定在中間 (5, 20)，並嘗試放大
        # 注意：如果預設字體太小，模型會很難練。這裡手動重複繪製增加粗度。
        display_text = label_text.strip()
        d.text((5, 20), display_text, fill=255)
        # 稍微重疊繪製來模擬粗體 (Bold)
        d.text((6, 20), display_text, fill=255) 
        
        img_array = np.array(img).astype(np.float32) / 255.0
        texts.append(label_text)
        images.append(img_array)
    return texts, torch.tensor(np.array(images))

def img_to_patches(img):
    B, H, W = img.shape
    p = patch_size
    patches = img.view(B, H//p, p, W//p, p).permute(0, 1, 3, 2, 4).contiguous()
    return patches.view(B, -1, p*p)

def patches_to_img(patches):
    B, _, _ = patches.shape
    p = patch_size
    grid = img_size // p
    img = patches.view(B, grid, grid, p, p).permute(0, 1, 3, 2, 4).contiguous()
    return img.view(B, img_size, img_size)

# --- 3. 模型架構 ---
class Number2ImgTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, text_len + num_patches, n_embd))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_embd, nhead=n_head, dim_feedforward=n_embd*4, 
            batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
        self.out_head = nn.Linear(n_embd, patch_dim)

    def forward(self, text_idx):
        B = text_idx.size(0)
        t_emb = self.token_emb(text_idx)
        p_query = torch.zeros(B, num_patches, n_embd, device=device)
        
        x = torch.cat((t_emb, p_query), dim=1)
        x = x + self.pos_emb
        x = self.transformer(x)
        
        img_out = self.out_head(x[:, text_len:, :])
        return torch.sigmoid(img_out)

# --- 4. 訓練 ---
model = Number2ImgTransformer().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

print(f"使用設備: {device} | 訓練目標：純數字生成")

for i in range(max_iters + 1):
    texts, imgs = generate_batch(batch_size)
    text_in = torch.tensor([[stoi[c] for c in t] for t in texts]).to(device)
    target_patches = img_to_patches(imgs.to(device))
    
    pred_patches = model(text_in)
    loss = F.binary_cross_entropy(pred_patches, target_patches)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if i % 500 == 0:
        print(f"迭代 {i:4d}, Loss: {loss.item():.4f}")
        # 每 500 次存一張測試圖 (固定測試數字 12345)
        with torch.no_grad():
            test_idx = torch.tensor([[stoi[c] for c in "12345 ".ljust(text_len)]]).to(device)
            res = patches_to_img(model(test_idx)).cpu().numpy()[0]
            Image.fromarray((res * 255).astype(np.uint8)).save(f"num_iter_{i}.png")

# --- 5. 測試生成 ---
def test_number(n_str):
    n_str = n_str[:text_len].ljust(text_len)
    idx = torch.tensor([[stoi.get(c, stoi[' ']) for c in n_str]]).to(device)
    with torch.no_grad():
        out = patches_to_img(model(idx)).cpu().numpy()[0]
        Image.fromarray((out * 255).astype(np.uint8)).save(f"result_{n_str.strip()}.png")
        print(f"已生成: result_{n_str.strip()}.png")

test_number("987654")
test_number("0")