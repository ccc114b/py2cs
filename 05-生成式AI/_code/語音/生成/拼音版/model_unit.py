import torch
import torch.nn as nn

class UnitGenerator(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, mel_channels=80):
        super(UnitGenerator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # 嚴格遵循 index 順序的架構
        self.network = nn.Sequential(
            nn.Linear(embedding_dim, 512),       # index 0
            nn.ReLU(),                           # index 1
            nn.Dropout(0.1),                     # index 2
            nn.Linear(512, 1024),                # index 3
            nn.ReLU(),                           # index 4
            # 【修改點 1】 這裡要改為 40 * mel_channels (1024 -> 3200)
            nn.Linear(1024, 40 * mel_channels)   # index 5
        )

    def forward(self, char_id):
        x = self.embedding(char_id)
        mel_flat = self.network(x)
        # 【修改點 2】 輸出的 view 也要改為 40 幀
        # 從 [Batch, 3200] 轉為 [Batch, 40, 80]
        return mel_flat.view(-1, 40, 80)