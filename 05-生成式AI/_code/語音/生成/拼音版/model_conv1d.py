import torch
import torch.nn as nn
from model_unit import UnitGenerator

class NeuralLinker(nn.Module):
    def __init__(self, mel_channels=80):
        super(NeuralLinker, self).__init__()
        # 使用極小的卷積核，只修正微小的雜訊
        self.refiner = nn.Sequential(
            nn.Conv1d(mel_channels, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2), # 使用 LeakyReLU 保留更多特徵
            nn.Conv1d(128, mel_channels, kernel_size=3, padding=1)
        )

    def forward(self, x, alpha=0.05):
        # x: [Batch, Frames, 80]
        identity = x.transpose(1, 2) 
        
        # 預測修正殘差
        delta = self.refiner(identity)
        
        # 【核心改進】Alpha 控制干預強度
        # 我們讓 Model B 的影響力降到極低
        out = identity + alpha * delta
        return out.transpose(1, 2)