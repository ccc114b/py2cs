import torch
import torch.nn as nn
from torch.nn import functional as F
from PIL import Image, ImageDraw
import numpy as np
import string
import random

# --- 1. 超參數與環境設定 ---
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
n_embd = 128
n_head = 8
n_layer = 6
max_iters = 5000      # 增加迭代次數以應付更多樣的字元組合
batch_size = 32
img_size = 64
patch_size = 8        # 8x8 patch
num_patches = (img_size // patch_size) ** 2 
patch_dim = patch_size * patch_size 
text_len = 12         # 增加最大字串長度

# 定義詞彙表：包含大小寫字母、數字、標點符號與空格
# all_chars = string.ascii_letters + string.digits + string.punctuation + " "
all_chars = string.ascii_letters + string.digits + string.punctuation + " "
stoi = { ch:i for i,ch in enumerate(all_chars) }
itos = { i:ch for i,ch in enumerate(all_chars) }
vocab_size = len(all_chars)

# --- 2. 增強版數據生成器 ---
def generate_batch(batch_size):
    texts = []
    images = []
    for _ in range(batch_size):
        # 隨機決定字串長度並隨機挑選字元
        curr_len = random.randint(3, text_len)
        random_str = ''.join(random.choice(all_chars[:-1]) for _ in range(curr_len))
        label_text = random_str.ljust(text_len) # 補齊長度以符合固定 Tensor 形狀
        
        # 繪製圖片
        img = Image.new('L', (img_size, img_size), color=0)
        d = ImageDraw.Draw(img)
        
        # 簡單的動態座標：讓文字在中間區域隨機晃動，增加模型泛化力
        x_pos = random.randint(2, 10)
        y_pos = random.randint(15, 25)
        d.text((x_pos, y_pos), label_text.strip(), fill=255)
        
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

# --- 3. 多模態 Transformer (ViT-based) ---
class ArbitraryText2Img(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        # 位置編碼：涵蓋文字 Token 與 影像 Patch Token
        self.pos_emb = nn.Parameter(torch.zeros(1, text_len + num_patches, n_embd))
        
        # Transformer 核心
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_embd, nhead=n_head, dim_feedforward=n_embd*4, 
            batch_first=True, activation='gelu', dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
        
        # 輸出層：預測每個 Patch 的像素值
        self.out_head = nn.Linear(n_embd, patch_dim)

    def forward(self, text_idx):
        B = text_idx.size(0)
        
        # 1. 嵌入文字
        t_emb = self.token_emb(text_idx) # (B, text_len, n_embd)
        
        # 2. 建立影像 Query (模型將在此處「畫出」圖片)
        p_query = torch.zeros(B, num_patches, n_embd, device=device)
        
        # 3. 合併並加入位置資訊
        x = torch.cat((t_emb, p_query), dim=1)
        x = x + self.pos_emb
        
        # 4. 透過 Transformer 進行注意力運算（文字會引導影像 Query 的生成）
        x = self.transformer(x)
        
        # 5. 取出影像部分並映射至像素空間
        img_out = self.out_head(x[:, text_len:, :])
        return torch.sigmoid(img_out)

# --- 4. 訓練程序 ---
model = ArbitraryText2Img().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4)

print(f"設備: {device} | 詞彙量: {vocab_size} | 訓練開始...")

for i in range(max_iters + 1):
    texts, imgs = generate_batch(batch_size)
    text_in = torch.tensor([[stoi[c] for c in t] for t in texts]).to(device)
    target_patches = img_to_patches(imgs.to(device))
    
    pred_patches = model(text_in)
    
    # 計算像素層級的損失
    loss = F.binary_cross_entropy(pred_patches, target_patches)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if i % 100 == 0:
        print(f"迭代 {i}/{max_iters}, Loss: {loss.item():.4f}")
        # 測試生成一張隨機內容
        test_val = "".join(random.choice(string.ascii_letters) for _ in range(5)).ljust(text_len)
        test_idx = torch.tensor([[stoi[c] for c in test_val]]).to(device)
        with torch.no_grad():
            res = patches_to_img(model(test_idx)).cpu().numpy()[0]
            Image.fromarray((res * 255).astype(np.uint8)).save(f"img/iter_{i}.png")

# --- 5. 互動生成測試 ---
def generate_custom_text(user_str):
    print(f"正在為 '{user_str}' 生成影像...")
    user_str = user_str[:text_len].ljust(text_len)
    idx = torch.tensor([[stoi.get(c, stoi[' ']) for c in user_str]]).to(device)
    with torch.no_grad():
        out = patches_to_img(model(idx)).cpu().numpy()[0]
        Image.fromarray((out * 255).astype(np.uint8)).save("img/custom_output.png")
        print("存檔成功: img/custom_output.png")

# 訓練結束後測試
generate_custom_text("Hello! AI")
generate_custom_text("DeepSeek 1")