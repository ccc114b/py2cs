import torch, torch.nn as nn, json
from PIL import Image
import numpy as np

# 載入配置
with open("dynamic_config.json", "r", encoding="utf-8") as f:
    cfg = json.load(f)

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
chars = cfg["chars"]
stoi = { ch:i for i,ch in enumerate(chars) }
text_len = cfg["text_len"]

# 模型結構需對應
class DynamicT2I(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(len(chars), cfg["n_embd"])
        self.pos_emb = nn.Parameter(torch.zeros(1, text_len + 16, cfg["n_embd"]))
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=cfg["n_embd"], nhead=cfg["n_head"], batch_first=True), num_layers=cfg["n_layer"])
        self.ln_f = nn.LayerNorm(cfg["n_embd"])
        self.patch_head = nn.Linear(cfg["n_embd"], 64)

    def forward(self, idx):
        B = idx.size(0)
        x = torch.cat((self.token_emb(idx), torch.zeros(B, 16, cfg["n_embd"], device=device)), dim=1)
        x = self.transformer(x + self.pos_emb)
        return torch.sigmoid(self.patch_head(self.ln_f(x[:, text_len:, :])))

model = DynamicT2I().to(device)
model.load_state_dict(torch.load("dynamic_t2i.pth", map_location=device))
model.eval()

def predict(cmd):
    print(f"輸入指令: {cmd}")
    idx = torch.tensor([[stoi.get(c, stoi[' ']) for c in cmd[:text_len].ljust(text_len)]]).to(device)
    with torch.no_grad():
        out = model(idx)
        img = out.view(1, 2, 8, 8, 8).permute(0, 1, 3, 2, 4).reshape(16, 64).cpu().numpy()
        res = Image.fromarray((img * 255).astype(np.uint8)).resize((256, 64), resample=Image.NEAREST)
        res.show()
        res.save("test_result.png")

if __name__ == "__main__":
    predict("麻煩寫出 731")
    predict("請畫數字 5")