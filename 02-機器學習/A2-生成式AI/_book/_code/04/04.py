import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 設定隨機種子以利復現
torch.manual_seed(42)
np.random.seed(42)

class DiffusionModel(nn.Module):
    """
    簡易擴散模型實作
    使用一個簡單的多層感知器 (MLP) 來預測雜訊
    """
    def __init__(self, input_dim=2, hidden_dim=64):
        super(DiffusionModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim), # +1 用於時間編碼 t
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x_t, t):
        # 將時間步 t 轉為與 x_t 相同的維度並合併
        t_input = t.view(-1, 1).float()
        combined = torch.cat([x_t, t_input], dim=1)
        return self.network(combined)

def train_diffusion():
    # 1. 參數設定
    T = 100  # 總時間步
    beta = torch.linspace(1e-4, 0.02, T) # 變異數排程 (Variance Schedule)
    alpha = 1.0 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    
    # 2. 生成模擬數據：原始數據分佈 q(x_0) (S型分佈)
    n_samples = 1000
    theta = torch.linspace(-1.5 * np.pi, 1.5 * np.pi, n_samples)
    x0 = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1) * 0.5
    x0 += torch.randn_like(x0) * 0.05 # 加入微小雜訊

    # 3. 初始化模型與優化器
    model = DiffusionModel(input_dim=2)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    mse_loss = nn.MSELoss()

    # 4. 訓練迴圈
    epochs = 500
    batch_size = 128
    
    for epoch in range(epochs):
        # 隨機選取樣本與時間步
        indices = torch.randint(0, n_samples, (batch_size,))
        batch_x0 = x0[indices]
        t = torch.randint(0, T, (batch_size,))
        
        # 重參數化技巧 (Reparameterization Trick)
        # x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
        epsilon = torch.randn_like(batch_x0)
        a_bar_t = alpha_bar[t].view(-1, 1)
        
        xt = torch.sqrt(a_bar_t) * batch_x0 + torch.sqrt(1 - a_bar_t) * epsilon
        
        # 預測雜訊並計算損失
        predicted_noise = model(xt, t)
        loss = mse_loss(predicted_noise, epsilon)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

    # 5. 逆向採樣過程 (Reverse Sampling)
    model.eval()
    with torch.no_grad():
        # 從純高斯雜訊 x_T 開始
        cur_x = torch.randn(500, 2)
        samples = [cur_x.numpy()]
        
        for t_step in reversed(range(T)):
            t_tensor = torch.full((500,), t_step)
            z = torch.randn_like(cur_x) if t_step > 0 else 0
            
            # 預測雜訊
            eps_theta = model(cur_x, t_tensor)
            
            # 根據逆向分佈公式計算 x_{t-1}
            # x_{t-1} = 1/sqrt(alpha_t) * (x_t - beta_t/sqrt(1-alpha_bar_t) * eps_theta) + sigma_t * z
            alpha_t = alpha[t_step]
            a_bar_t = alpha_bar[t_step]
            beta_t = beta[t_step]
            
            mean = (1 / torch.sqrt(alpha_t)) * (cur_x - (beta_t / torch.sqrt(1 - a_bar_t)) * eps_theta)
            cur_x = mean + torch.sqrt(beta_t) * z
            
            if t_step % 20 == 0 or t_step == 0:
                samples.append(cur_x.numpy())

    # 6. 結果視覺化
    plt.figure(figsize=(15, 5))
    titles = ['Initial Noise', 't=80', 't=60', 't=40', 't=20', 'Final (t=0)', 'Ground Truth']
    display_indices = [0, 1, 2, 3, 4, 5]
    
    for i, idx in enumerate(display_indices):
        plt.subplot(1, 7, i+1)
        plt.scatter(samples[idx][:, 0], samples[idx][:, 1], alpha=0.5, s=10)
        plt.title(titles[i])
        plt.axis('equal')
        
    plt.subplot(1, 7, 7)
    plt.scatter(x0[:, 0], x0[:, 1], color='red', alpha=0.5, s=10)
    plt.title(titles[6])
    plt.axis('equal')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train_diffusion()