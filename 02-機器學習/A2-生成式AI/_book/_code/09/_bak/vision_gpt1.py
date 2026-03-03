import torch
import torch.nn as nn
from torch.nn import functional as F
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# --- 超參數與規劃 ---
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
n_embd = 128
n_head = 4
n_layer = 4
block_size = 64 # 總序列長度 (文字 tokens + 圖片 patches)

# 圖片規劃：64x64 像素
img_size = 64
patch_size = 16 # 每個小方塊 16x16 像素
num_patches = (img_size // patch_size) ** 2 # 總共會有 16 個 patches
patch_dim = patch_size * patch_size # 每個 patch 的向量維度 (16*16=256)

# 1. 數據生成器模組：產生「文字 -> 圖片」配對
def generate_fake_data(batch_size=32):
    texts = []
    images = []
    for _ in range(batch_size):
        age = np.random.randint(0, 100)
        label_text = f"age={age}"
        
        # 繪製 64x64 灰階圖
        img = Image.new('L', (img_size, img_size), color=0)
        d = ImageDraw.Draw(img)
        # 這裡簡單畫出文字 (實際使用建議載入 ttf 字體)
        d.text((5, 20), label_text, fill=255)
        
        img_array = np.array(img).astype(np.float32) / 255.0 # 歸一化到 0~1
        
        texts.append(label_text)
        images.append(img_array)
        
    return texts, torch.tensor(np.array(images))

# 2. 影像切片工具：將圖片轉為 Patches (ViT 核心概念)
def img_to_patches(img):
    # img shape: (B, H, W) -> (B, num_patches, patch_dim)
    B, H, W = img.shape
    p = patch_size
    patches = img.view(B, H//p, p, W//p, p).permute(0, 1, 3, 2, 4).contiguous()
    return patches.view(B, -1, p*p)

def patches_to_img(patches):
    # 將 patches 還原回圖片
    B, _, _ = patches.shape
    p = patch_size
    grid = img_size // p
    img = patches.view(B, grid, grid, p, p).permute(0, 1, 3, 2, 4).contiguous()
    return img.view(B, img_size, img_size)

# 3. 模型定義：Text-to-Patch Transformer
class TextToImageViT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # 文字嵌入
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        # 影像 Patch 嵌入 (將 256 維降到 n_embd)
        self.patch_projection = nn.Linear(patch_dim, n_embd)
        # 位置編碼
        self.pos_embedding = nn.Parameter(torch.zeros(1, block_size, n_embd))
        
        # Transformer Blocks (同 CharGPT)
        self.blocks = nn.Sequential(*[
            nn.TransformerEncoderLayer(d_model=n_embd, nhead=n_head, batch_first=True)
            for _ in range(n_layer)
        ])
        
        # 輸出層：將向量轉回像素 Patch (256 維)
        self.head = nn.Linear(n_embd, patch_dim)

    def forward(self, text_idx, patches):
        # text_idx: (B, text_len)
        # patches: (B, num_patches, patch_dim)
        
        # 1. 準備 Embedding 序列
        text_emb = self.token_embedding(text_idx) # (B, T, C)
        patch_emb = self.patch_projection(patches) # (B, P, C)
        
        x = torch.cat((text_emb, patch_emb), dim=1) # 合併文字與影像序列
        x = x + self.pos_embedding[:, :x.size(1), :]
        
        # 2. Transformer 運算
        x = self.blocks(x)
        
        # 3. 只取影像部分並預測下一個 patch 的像素
        # 在這個簡化版本中，我們直接讓模型學會對應關係
        img_out = self.head(x[:, text_idx.size(1):, :]) 
        return img_out

# 4. 訓練與執行示範
# 建立詞彙表 (0-9, a-z, =, 標點)
chars = "0123456789abcdefghijklmnopqrstuvwxyz= "
stoi = { ch:i for i,ch in enumerate(chars) }
vocab_size = len(chars)

model = TextToImageViT(vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

print("開始訓練 Text-to-Image (ViT Patches) 模型...")

for i in range(500):
    texts, imgs = generate_fake_data(batch_size=16)
    imgs = imgs.to(device)
    
    # 文字轉為 index
    text_in = torch.tensor([[stoi.get(c, 37) for c in t.ljust(10)] for t in texts]).to(device)
    
    # 將圖片切成 patches 作為輸入 (自回歸時會需要遮罩，這裡簡化為監督學習)
    target_patches = img_to_patches(imgs) # (B, 16, 256)
    
    # Forward
    pred_patches = model(text_in, torch.zeros_like(target_patches))
    
    # 損失函數：使用 MSE (比較像素差異) 而非 CrossEntropy
    loss = F.mse_loss(pred_patches, target_patches)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if i % 100 == 0:
        print(f"迭代 {i}, Loss: {loss.item():.4f}")

# 5. 測試生成
test_text = "age=42".ljust(10)
text_idx = torch.tensor([[stoi.get(c, 37) for c in test_text]]).to(device)
with torch.no_grad():
    generated_patches = model(text_idx, torch.zeros((1, num_patches, patch_dim)).to(device))
    gen_img = patches_to_img(generated_patches).cpu().numpy()[0]

# 將結果存成圖檔
gen_img_rescaled = (gen_img * 255).astype(np.uint8)
Image.fromarray(gen_img_rescaled).save("output_test.png")
print("生成完成！請查看 output_test.png")