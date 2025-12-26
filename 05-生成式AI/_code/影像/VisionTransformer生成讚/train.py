import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import TinyDiT, get_device

# 超參數升級
BATCH_SIZE = 128
LR = 9e-4 # 降低學習率讓模型學得更細
TIMESTEPS = 1000
DEVICE = get_device()

def train():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = TinyDiT().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    
    betas = torch.linspace(1e-4, 0.02, TIMESTEPS).to(DEVICE)
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)

    print(f"訓練增強版 DiT... 設備: {DEVICE}")

    model.train()
    for epoch in range(20): # 建議跑 50 epoch 以上
        total_loss = 0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            # --- Label Dropout (CFG 關鍵) ---
            # 10% 的機率將標籤設為 10 (Null標籤)
            drop_mask = torch.rand(y.shape[0], device=DEVICE) < 0.1
            y = torch.where(drop_mask, torch.tensor(10, device=DEVICE), y)

            t = torch.randint(0, TIMESTEPS, (x.shape[0],), device=DEVICE).long()
            noise = torch.randn_like(x)
            a_t = alphas_cumprod[t].view(-1, 1, 1, 1)
            x_noisy = torch.sqrt(a_t) * x + torch.sqrt(1 - a_t) * noise
            
            pred = model(x_noisy, t, y)
            loss = F.mse_loss(pred, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch} | Loss: {total_loss/len(loader):.6f}")
        if epoch % 10 == 0:
            torch.save(model.state_dict(), "strong_dit_mnist.pth")

if __name__ == "__main__":
    train()