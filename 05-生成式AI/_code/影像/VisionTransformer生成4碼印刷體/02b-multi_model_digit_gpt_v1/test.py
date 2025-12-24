import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import json

# 載入設定與環境
with open("t2i_config.json", "r") as f:
    cfg = json.load(f)

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
chars = cfg["chars"]
stoi = { ch:i for i,ch in enumerate(chars) }
text_len = 10
img_w, img_h, patch_size = 64, 16, 8

class T2ITransformer(nn.Module):
    def __init__(self):
        super().__init__()
        # 確保這些參數與 model_config.json 中的內容一致
        self.token_embedding = nn.Embedding(len(chars), cfg["n_embd"])
        # 序列長度 = 文字長度(10) + 影像小塊數(16)
        self.position_embedding = nn.Parameter(torch.zeros(1, 10 + 16, cfg["n_embd"]))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg["n_embd"], nhead=cfg["n_head"], batch_first=True, activation='gelu'
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=cfg["n_layer"])
        
        # --- 補上缺失的這一層 ---
        self.ln_f = nn.LayerNorm(cfg["n_embd"]) 
        # ----------------------
        
        self.patch_head = nn.Linear(cfg["n_embd"], 64) # 64 是 patch_dim (8*8)

    def forward(self, idx):
        B = idx.size(0)
        tok_emb = self.token_embedding(idx)
        # 建立影像 Query 佔位符
        img_query = torch.zeros(B, 16, cfg["n_embd"], device=device)
        
        # 合併文字與影像序列
        x = torch.cat((tok_emb, img_query), dim=1) + self.position_embedding
        x = self.blocks(x)
        
        # --- 補上缺失的運算 ---
        x = self.ln_f(x) 
        # ----------------------
        
        # 只取出後半段 (影像部分) 的輸出
        return torch.sigmoid(self.patch_head(x[:, 10:, :]))

def patches_to_img(patches):
    B = patches.size(0)
    p = 8
    img = patches.view(B, 2, 8, p, p).permute(0, 1, 3, 2, 4).contiguous()
    return img.view(B, 16, 64)

# 載入模型
model = T2ITransformer().to(device)
model.load_state_dict(torch.load("t2i_gpt_model.pth", map_location=device))
model.eval()

def draw_text_command(cmd):
    # 格式化輸入
    print(f"執行指令: {cmd}")
    cmd_fixed = cmd.ljust(text_len)
    idx = torch.tensor([[stoi.get(c, stoi[' ']) for c in cmd_fixed]]).to(device)
    
    with torch.no_grad():
        pred_p = model(idx)
        img_np = patches_to_img(pred_p).cpu().numpy()[0]
    
    # 存檔
    res = Image.fromarray((img_np * 255).astype(np.uint8))
    res = res.resize((img_w*4, img_h*4), resample=Image.NEAREST)
    res.save("output_result.png")
    print("生成結果已儲存至 output_result.png")

if __name__ == "__main__":
    user_cmd = input("請輸入指令 (例如: 畫出 9836): ")
    draw_text_command(user_cmd)