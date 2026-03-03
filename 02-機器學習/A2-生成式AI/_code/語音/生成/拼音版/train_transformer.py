import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import librosa
import json
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from model_transformer import NeuralLinker
from model_unit import UnitGenerator # 同樣引入以載入權重

# 載入字典
with open("char_map.json", "r", encoding="utf-8") as f:
    char_map = json.load(f)

class EarlyStopping:
    def __init__(self, patience=40, delta=0.0001):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False

class FullSentenceDataset(Dataset):
    def __init__(self, metadata_path):
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.data = [line.strip().split('|') for line in f if '|' in line]
        self.mel_trans = torchaudio.transforms.MelSpectrogram(
            sample_rate=22050, n_fft=1024, hop_length=256, n_mels=80
        )

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        audio_path, py_text = self.data[idx]
        y, _ = librosa.load(audio_path, sr=22050)
        mel = self.mel_trans(torch.from_numpy(y).unsqueeze(0))
        # 正規化: log10 與 (x+2)/2
        log_mel = torch.log10(torch.clamp(mel, min=1e-5))
        norm_mel = (log_mel + 2.0) / 2.0
        tokens = [char_map[p] for p in py_text.split() if p in char_map]
        return torch.LongTensor(tokens), norm_mel.squeeze(0).T

def collate_fn(batch):
    texts, mels = zip(*batch)
    texts_pad = torch.nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=0)
    max_mel_len = max(m.size(0) for m in mels)
    mels_pad = torch.zeros(len(mels), max_mel_len, 80)
    for i, m in enumerate(mels):
        mels_pad[i, :m.size(0), :] = m
    return texts_pad, mels_pad

def train():
    # 1. 載入並冷凍 Model A
    unit_gen = UnitGenerator(vocab_size=len(char_map))
    unit_gen.load_state_dict(torch.load("unit_gen.pth", map_location='cpu'))
    unit_gen.eval()
    for p in unit_gen.parameters(): p.requires_grad = False

    # 2. 準備 Model B
    linker = NeuralLinker()
    optimizer = optim.Adam(linker.parameters(), lr=0.0005)
    criterion = nn.MSELoss()
    early_stop = EarlyStopping(patience=50)

    # 3. 資料準備
    full_dataset = FullSentenceDataset("metadata.txt")
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False, collate_fn=collate_fn)

    print("開始訓練 NeuralLinker (Model B - 40幀版)...")
    epoch = 0
    while not early_stop.early_stop:
        epoch += 1
        linker.train()
        train_loss = 0
        for text_seq, mel_target in train_loader:
            optimizer.zero_grad()
            with torch.no_grad():
                # Model A 產生 40 幀單元
                chunks = [unit_gen(text_seq[:, i]) for i in range(text_seq.size(1))]
                raw_mel = torch.cat(chunks, dim=1)
            
            # 長度縮放對齊全句錄音
            raw_mel = F.interpolate(raw_mel.transpose(1, 2), size=mel_target.size(1), mode='linear', align_corners=False).transpose(1, 2)
            
            output = linker(raw_mel)
            loss = criterion(output, mel_target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 驗證集檢查
        linker.eval()
        val_loss = 0
        with torch.no_grad():
            for text_seq, mel_target in val_loader:
                chunks = [unit_gen(text_seq[:, i]) for i in range(text_seq.size(1))]
                raw_mel = torch.cat(chunks, dim=1)
                raw_mel = F.interpolate(raw_mel.transpose(1, 2), size=mel_target.size(1), mode='linear', align_corners=False).transpose(1, 2)
                output = linker(raw_mel)
                val_loss += criterion(output, mel_target).item()
        
        avg_val = val_loss/len(val_loader)
        improved = early_stop(avg_val)
        
        if epoch % 5 == 0 or improved:
            msg = " (最佳模型更新)" if improved else ""
            print(f"Epoch {epoch}: Train Loss {train_loss/len(train_loader):.6f} | Val Loss {avg_val:.6f}{msg}")
        
        if improved:
            torch.save(linker.state_dict(), "linker_transformer.pth")

    print(f"訓練完成。最佳驗證 Loss: {early_stop.best_loss:.6f}")

if __name__ == "__main__":
    train()