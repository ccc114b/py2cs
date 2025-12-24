import torch
import torch.nn as nn

class UnitGenerator(nn.Module):
    """
    模型 A：發音單元生成器
    目標：輸入一個拼音 ID，輸出該拼音固定的發音特徵塊（20 幀）。
    """
    def __init__(self, vocab_size, embedding_dim=256, mel_channels=80):
        super(UnitGenerator, self).__init__()
        # 文字/拼音嵌入層
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # 多層線性網絡：將抽象的文字向量轉化為具體的音頻特徵
        # 輸出維度為 20(幀) * 80(梅爾維度) = 1600
        self.network = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 20 * mel_channels) 
        )

    def forward(self, char_id):
        # char_id: [Batch]
        x = self.embedding(char_id) # [Batch, embedding_dim]
        mel_flat = self.network(x)  # [Batch, 1600]
        # 轉換為 [Batch, 20幀, 80維]
        return mel_flat.view(-1, 20, 80)


class NeuralLinker(nn.Module):
    """
    模型 B：神經連接器 (Transformer)
    目標：接收拼接後的生硬頻譜，學習如何讓銜接處平滑並加入韻律感。
    """
    def __init__(self, mel_channels=80, nhead=8, num_layers=3):
        super(NeuralLinker, self).__init__()
        
        # Transformer 適合處理序列中不同位置的關聯
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=mel_channels, 
            nhead=nhead, 
            dim_feedforward=512, 
            batch_first=True, 
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 最終輸出修正層 (Post-net 的簡化版)
        self.post_net = nn.Sequential(
            nn.Linear(mel_channels, mel_channels),
            nn.Tanh(),
            nn.Linear(mel_channels, mel_channels)
        )

    def forward(self, combined_mel):
        # combined_mel: [Batch, Total_Frames, 80]
        # 1. 透過 Transformer 觀察全局資訊
        refined = self.transformer(combined_mel)
        # 2. 透過 Post-net 做微小的數值修正
        residual = self.post_net(refined)
        return refined + residual # 殘差結構，有助於保留原始發音