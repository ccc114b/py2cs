import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleTTS(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, mel_channels=80):
        super(SimpleTTS, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # LSTM 處理文字序列
        self.encoder = nn.LSTM(embedding_dim, 256, batch_first=True, bidirectional=True)
        # 轉為梅爾頻譜的維度
        self.decoder = nn.Linear(512, mel_channels)

    def forward(self, text_seq, target_len=None):
        # text_seq: [Batch, Text_Len]
        emb = self.embedding(text_seq) # [Batch, Text_Len, 256]
        enc_out, _ = self.encoder(emb) # [Batch, Text_Len, 512]
        mel_out = self.decoder(enc_out) # [Batch, Text_Len, 80]

        # 如果有給目標長度，就進行縮放 (Interpolation)
        if target_len is not None:
            # F.interpolate 需要 [Batch, Channels, Length]
            mel_out = mel_out.transpose(1, 2) # [Batch, 80, Text_Len]
            mel_out = F.interpolate(mel_out, size=target_len, mode='linear', align_corners=False)
            mel_out = mel_out.transpose(1, 2) # [Batch, target_len, 80]
            
        return mel_out