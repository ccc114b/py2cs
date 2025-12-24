import torch
import torch.nn as nn
from model_unit import UnitGenerator # 從獨立模組引入

class NeuralLinker(nn.Module):
    def __init__(self, mel_channels=80, nhead=8, num_layers=4):
        super(NeuralLinker, self).__init__()
        
        # Transformer 負責全局韻律與銜接
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=mel_channels, 
            nhead=nhead, 
            dim_feedforward=1024, 
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 殘差修正層：微調拼接點的頻譜數值
        self.refine_net = nn.Sequential(
            nn.Linear(mel_channels, mel_channels * 2),
            nn.ReLU(),
            nn.Linear(mel_channels * 2, mel_channels)
        )

    def forward(self, x):
        # x: Model A 拼接後的頻譜 [Batch, Total_Frames, 80]
        h = self.transformer(x)
        delta = self.refine_net(h)
        # 輸出：基礎頻譜 + 10% 的神經修正 (保持 Model A 的清晰度)
        return h + 0.1 * delta