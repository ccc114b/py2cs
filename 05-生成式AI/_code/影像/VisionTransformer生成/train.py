import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from model import TinyDiT, get_device

# 超參數優化
BATCH_SIZE = 256  # 增大 Batch
EPOCHS = 50       # 減少次數
LR = 1e-3         # 稍微調大學習率
TIMESTEPS = 1000
DEVICE = get_device()

betas = torch.linspace(1e-4, 0.02, TIMESTEPS).to(DEVICE)
alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)

def train():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    loader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform), 
                                         batch_size=BATCH_SIZE, shuffle=True)

    model = TinyDiT().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    model.train()
    for epoch in range(EPOCHS):
        for x, _ in loader:
            x = x.to(DEVICE)
            t = torch.randint(0, TIMESTEPS, (x.shape[0],), device=DEVICE).long()
            noise = torch.randn_like(x)
            
            # 加噪
            a_t = alphas_cumprod[t].view(-1, 1, 1, 1)
            x_noisy = torch.sqrt(a_t) * x + torch.sqrt(1 - a_t) * noise
            
            # 預測
            pred = model(x_noisy, t)
            loss = F.mse_loss(pred, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        print(f"Epoch {epoch} | Loss: {loss.item():.6f}")

    torch.save(model.state_dict(), "tiny_dit_mnist.pth")

if __name__ == "__main__":
    train()