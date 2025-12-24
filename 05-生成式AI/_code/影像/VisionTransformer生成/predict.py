import torch
from model import TinyDiT, get_device
import matplotlib.pyplot as plt

TIMESTEPS = 1000
SKIP = 10 # 每 10 步跳一次
DEVICE = get_device()

betas = torch.linspace(1e-4, 0.02, TIMESTEPS).to(DEVICE)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)

@torch.no_grad()
def sample():
    model = TinyDiT().to(DEVICE)
    model.load_state_dict(torch.load("tiny_dit_mnist.pth", map_location=DEVICE))
    model.eval()

    img = torch.randn((16, 1, 28, 28), device=DEVICE)

    # 快速採樣 (只跑 100 步)
    for i in reversed(range(0, TIMESTEPS, SKIP)):
        t = torch.full((16,), i, device=DEVICE, dtype=torch.long)
        pred_noise = model(img, t)
        
        # 這裡簡化計算，快速去噪
        a_t = alphas_cumprod[i]
        beta_t = betas[i]
        img = (img - torch.sqrt(1 - a_t) * pred_noise) / torch.sqrt(a_t)
        
        if i > 0:
            img = torch.sqrt(alphas_cumprod[i-SKIP]) * img + torch.sqrt(1 - alphas_cumprod[i-SKIP]) * torch.randn_like(img)

    plt.figure(figsize=(4,4))
    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.imshow(img[i].cpu().squeeze(), cmap='gray')
        plt.axis('off')
    plt.show()

if __name__ == "__main__":
    sample()