import torch
import torch.nn as nn
from torch.nn import functional as F
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# --- 1. 超參數調整 (Scaling Up) ---
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
n_embd = 32           # 增加維度
n_head = 8            # 增加頭數
n_layer = 6           # 增加層數
max_iters = 2000      # 顯著增加訓練次數
batch_size = 32
img_size = 64
patch_size = 8        # 縮小 Patch，讓模型看得更細 (8x8 patch)
num_patches = (img_size // patch_size) ** 2 
patch_dim = patch_size * patch_size 
text_len = 10         # 固定文字輸入長度

# 2. 改進的數據生成器
def generate_batch(batch_size):
    texts = []
    images = []
    for _ in range(batch_size):
        age = np.random.randint(0, 100)
        label_text = f"age={age}".ljust(text_len)
        img = Image.new('L', (img_size, img_size), color=0)
        d = ImageDraw.Draw(img)
        # 繪製較大的文字，讓模型容易學習
        d.text((5, 20), label_text, fill=255)
        
        img_array = np.array(img).astype(np.float32) / 255.0
        texts.append(label_text)
        images.append(img_array)
    return texts, torch.tensor(np.array(images))

def img_to_patches(img):
    B, H, W = img.shape
    p = patch_size
    # (B, 8, 8, 8, 8) -> (B, 64, 64)
    patches = img.view(B, H//p, p, W//p, p).permute(0, 1, 3, 2, 4).contiguous()
    return patches.view(B, -1, p*p)

def patches_to_img(patches):
    B, _, _ = patches.shape
    p = patch_size
    grid = img_size // p
    img = patches.view(B, grid, grid, p, p).permute(0, 1, 3, 2, 4).contiguous()
    return img.view(B, img_size, img_size)

# 3. Text-to-Image Transformer
class T2ITransformer(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, text_len + num_patches, n_embd))
        
        # 使用標準的 Transformer Decoder 結構
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=n_embd, nhead=n_head, dim_feedforward=n_embd*4, 
            batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=n_layer)
        
        self.out_head = nn.Linear(n_embd, patch_dim)

    def forward(self, text_idx):
        B = text_idx.size(0)
        
        # 這裡我們只傳入文字，影像部分用零向量作為 Query
        t_emb = self.token_emb(text_idx)
        p_query = torch.zeros(B, num_patches, n_embd, device=device)
        
        x = torch.cat((t_emb, p_query), dim=1)
        x = x + self.pos_emb
        
        x = self.transformer(x)
        
        # 只取出對應影像 patch 的部分
        img_logits = self.out_head(x[:, text_len:, :])
        return torch.sigmoid(img_logits) # 使用 Sigmoid 配合 BCE Loss

# 4. 訓練邏輯
chars = "0123456789abcdefghijklmnopqrstuvwxyz= "
stoi = { ch:i for i,ch in enumerate(chars) }
model = T2ITransformer(len(chars)).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

print(f"啟動模型: {device} | 參數量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

for i in range(max_iters + 1):
    texts, imgs = generate_batch(batch_size)
    text_in = torch.tensor([[stoi.get(c, 37) for c in t] for t in texts]).to(device)
    target_patches = img_to_patches(imgs.to(device))
    
    pred_patches = model(text_in)
    
    # 改用 BCELoss，對黑白對比效果更好
    loss = F.binary_cross_entropy(pred_patches, target_patches)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if i %100 == 0:
        print(f"進度 {i}/{max_iters}, Loss: {loss.item():.4f}")
        
        # 每 100 次存一張圖看看進度
        with torch.no_grad():
            sample_text = "age=56".ljust(text_len)
            sample_idx = torch.tensor([[stoi.get(c, 37) for c in sample_text]]).to(device)
            res_patches = model(sample_idx)
            res_img = patches_to_img(res_patches).cpu().numpy()[0]
            Image.fromarray((res_img * 255).astype(np.uint8)).save(f"img/progress_{i}.png")

# 5. 最終測試
test_str = "age=56".ljust(text_len)
print(f"最終生成測試: {test_str}")
test_idx = torch.tensor([[stoi.get(c, 37) for c in test_str]]).to(device)
with torch.no_grad():
    final_patches = model(test_idx)
    final_img = patches_to_img(final_patches).cpu().numpy()[0]
    Image.fromarray((final_img * 255).astype(np.uint8)).save("img/final_output.png")