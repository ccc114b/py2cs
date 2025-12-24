import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 定義生成器 (Generator)
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim)
        )
    
    def forward(self, z):
        return self.net(z)

# 定義判別器 (Discriminator)
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # 輸出機率
        )
    
    def forward(self, x):
        return self.net(x)

# 超參數設定
latent_dim = 5      # 雜訊 z 的維度
data_dim = 1        # 真實數據維度 (1D)
lr = 0.001          # 學習率
epochs = 5000       # 訓練代數
batch_size = 64     # 批次大小

# 實例化網路與優化器
G = Generator(latent_dim, data_dim)
D = Discriminator(data_dim)
optimizer_G = optim.Adam(G.parameters(), lr=lr)
optimizer_D = optim.Adam(D.parameters(), lr=lr)
criterion = nn.BCELoss() # 二元交叉熵損失

# 訓練循環
for epoch in range(epochs):
    # --- 1. 訓練判別器 ---
    # 真實數據: 服從 N(4, 1.5) 的分佈
    real_data = torch.randn(batch_size, data_dim) * 1.5 + 4.0
    real_labels = torch.ones(batch_size, 1)
    fake_labels = torch.zeros(batch_size, 1)
    
    # 計算真實數據的 Loss
    outputs_real = D(real_data)
    loss_real = criterion(outputs_real, real_labels)
    
    # 生成偽造數據
    z = torch.randn(batch_size, latent_dim)
    fake_data = G(z)
    outputs_fake = D(fake_data.detach()) # detach 避免更新到 G
    loss_fake = criterion(outputs_fake, fake_labels)
    
    loss_D = loss_real + loss_fake
    optimizer_D.zero_grad()
    loss_D.backward()
    optimizer_D.step()
    
    # --- 2. 訓練生成器 ---
    z = torch.randn(batch_size, latent_dim)
    fake_data = G(z)
    outputs_G = D(fake_data)
    
    # 生成器希望 D 把偽造數據判斷為「真」(1)
    loss_G = criterion(outputs_G, real_labels)
    
    optimizer_G.zero_grad()
    loss_G.backward()
    optimizer_G.step()
    
    if (epoch+1) % 1000 == 0:
        print(f"Epoch [{epoch+1}/{epochs}] | Loss D: {loss_D.item():.4f} | Loss G: {loss_G.item():.4f}")

# 視覺化結果
with torch.no_grad():
    test_z = torch.randn(1000, latent_dim)
    generated_data = G(test_z).numpy()
    real_sample = (torch.randn(1000, data_dim) * 1.5 + 4.0).numpy()

plt.hist(real_sample, bins=30, alpha=0.5, label='Real Data', density=True)
plt.hist(generated_data, bins=30, alpha=0.5, label='Generated Data', density=True)
plt.legend()
plt.title("GAN Data Distribution Learning")
plt.show()