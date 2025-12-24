import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import json
import random

# 1. 載入設定與初始化設備
with open("model_config.json", "r") as f:
    config = json.load(f)

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
chars = config["chars"]
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

# 2. 定義相同的模型結構 (必須與訓練時一致)
class T2ITransformer(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, config["n_embd"])
        num_p = (config["img_w"] // config["patch_size"]) * (config["img_h"] // config["patch_size"])
        self.pos_emb = nn.Parameter(torch.zeros(1, config["text_len"] + num_p, config["n_embd"]))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=config["n_embd"], nhead=config["n_head"], batch_first=True, activation='gelu'),
            num_layers=config["n_layer"]
        )
        self.out_head = nn.Linear(config["n_embd"], config["patch_size"]**2)

    def forward(self, text_idx):
        B = text_idx.size(0)
        t_emb = self.token_emb(text_idx)
        p_query = torch.zeros(B, (config["img_w"] // config["patch_size"]) * (config["img_h"] // config["patch_size"]), config["n_embd"], device=device)
        x = torch.cat((t_emb, p_query), dim=1) + self.pos_emb
        x = self.transformer(x)
        return torch.sigmoid(self.out_head(x[:, config["text_len"]:, :]))

def patches_to_img(patches):
    B, _, _ = patches.shape
    p = config["patch_size"]
    grid_w, grid_h = config["img_w"] // p, config["img_h"] // p
    img = patches.view(B, grid_h, grid_w, p, p).permute(0, 1, 3, 2, 4).contiguous()
    return img.view(B, config["img_h"], config["img_w"])

# 3. 載入權重
model = T2ITransformer(len(chars)).to(device)
model.load_state_dict(torch.load("num_t2i_model.pth", map_location=device))
model.eval() # 設定為評估模式

print("模型載入成功，開始產生 10 組檢驗結果...")

# 4. 產生 10 組隨機數字進行預測
results = []
for i in range(10):
    test_num = str(random.randint(0, 9999))
    print(f"[{i+1}/10] 正在為數字 '{test_num}' 繪圖...")
    
    # 文字轉 Tensor
    text_fixed = test_num.ljust(config["text_len"])
    idx = torch.tensor([[stoi.get(c, stoi[' ']) for c in text_fixed]]).to(device)
    
    with torch.no_grad():
        pred_patches = model(idx)
        img_np = patches_to_img(pred_patches).cpu().numpy()[0]
        
    # 將結果轉為 PIL 圖片並放大方便觀察
    img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
    img_pil = img_pil.resize((config["img_w"]*4, config["img_h"]*4), resample=Image.NEAREST)
    results.append(img_pil)

# 5. 合併成一張大圖展示
combined = Image.new('L', (config["img_w"]*4, config["img_h"]*4*10))
for i, img in enumerate(results):
    combined.paste(img, (0, i * config["img_h"] * 4))

combined.save("verification_results.png")
print("\n檢驗完成！請查看 verification_results.png，裡面包含 10 組隨機數字的生成結果。")