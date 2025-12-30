import torch
from model import TinyDiT, get_device
import matplotlib.pyplot as plt

TIMESTEPS = 1000
DEVICE = get_device()
CFG_SCALE = 3.0  # 導引強度，通常 3.0~7.5 效果最好

@torch.no_grad()
def sample(target_digit=5):
    model = TinyDiT().to(DEVICE)
    model.load_state_dict(torch.load("strong_dit_mnist.pth", map_location=DEVICE))
    model.eval()

    # 準備有條件標籤與空標籤
    y_cond = torch.full((16,), target_digit, device=DEVICE, dtype=torch.long)
    y_null = torch.full((16,), 10, device=DEVICE, dtype=torch.long)
    
    img = torch.randn((16, 1, 28, 28), device=DEVICE)
    betas = torch.linspace(1e-4, 0.02, TIMESTEPS).to(DEVICE)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    for i in reversed(range(TIMESTEPS)):
        t = torch.full((16,), i, device=DEVICE, dtype=torch.long)
        
        # --- CFG 採樣核心 ---
        # 同時預測有條件與無條件的雜訊
        noise_pred_cond = model(img, t, y_cond)
        noise_pred_uncond = model(img, t, y_null)
        
        # 這裡就是關鍵公式：向「更有特徵」的方向推進
        noise_pred = noise_pred_uncond + CFG_SCALE * (noise_pred_cond - noise_pred_uncond)
        
        # 標準 DDPM 步驟
        a_t = alphas[i]
        a_bar_t = alphas_cumprod[i]
        beta_t = betas[i]
        
        if i > 0:
            noise = torch.randn_like(img)
        else:
            noise = 0
            
        img = (1 / torch.sqrt(a_t)) * (img - ((1 - a_t) / torch.sqrt(1 - a_bar_t)) * noise_pred) + torch.sqrt(beta_t) * noise

    img = (img.clamp(-1, 1) + 1) / 2
    plt.figure(figsize=(4,4))
    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.imshow(img[i].cpu().squeeze(), cmap='gray')
        plt.axis('off')
    plt.show()

if __name__ == "__main__":
    for digit in range(0,10):
        sample(target_digit=digit)
