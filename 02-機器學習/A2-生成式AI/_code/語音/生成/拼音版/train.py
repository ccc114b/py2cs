import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import librosa
import json
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from model import NeuralLinker
from model_unit import UnitGenerator

# 1. 載入字元映射 (必須與 Model A 訓練時一致)
with open("char_map.json", "r", encoding="utf-8") as f:
    char_map = json.load(f)

# 2. 早停機制與最佳權重保存
class EarlyStopping:
    def __init__(self, patience=30, delta=0.0001):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            return True # 進步了，應該存檔
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False

# 3. 資料集定義 (全句錄音)
class FullSentenceDataset(Dataset):
    def __init__(self, metadata_path):
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.data = [line.strip().split('|') for line in f if '|' in line]
        
        # 頻譜參數需與 Model A 一致
        self.mel_trans = torchaudio.transforms.MelSpectrogram(
            sample_rate=22050, n_fft=1024, hop_length=256, n_mels=80
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_path, py_text = self.data[idx]
        # 載入錄音
        y, _ = librosa.load(audio_path, sr=22050)
        waveform = torch.from_numpy(y).unsqueeze(0)
        
        # 轉梅爾頻譜並正規化 (與 Model A 一致的 log10 正規化)
        mel = self.mel_trans(waveform)
        log_mel = torch.log10(torch.clamp(mel, min=1e-5))
        norm_mel = (log_mel + 2.0) / 2.0
        
        # 文字轉拼音 ID
        tokens = [char_map[p] for p in py_text.split() if p in char_map]
        return torch.LongTensor(tokens), norm_mel.squeeze(0).T

# 4. Batch 處理 (補零對齊)
def collate_fn(batch):
    texts, mels = zip(*batch)
    texts_pad = torch.nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=0)
    max_mel_len = max(m.size(0) for m in mels)
    mels_pad = torch.zeros(len(mels), max_mel_len, 80)
    for i, m in enumerate(mels):
        mels_pad[i, :m.size(0), :] = m
    return texts_pad, mels_pad

# 5. 訓練主程式
def train():
    # A. 載入並冷凍 Model A (40 幀版)
    unit_gen = UnitGenerator(vocab_size=len(char_map))
    if not os.path.exists("unit_gen.pth"):
        print("錯誤：找不到 unit_gen.pth，請先訓練 Model A。")
        return
    unit_gen.load_state_dict(torch.load("unit_gen.pth", map_location='cpu'))
    unit_gen.eval()
    for param in unit_gen.parameters():
        param.requires_grad = False

    # B. 初始化 Model B (卷積平滑器)
    linker = NeuralLinker()
    optimizer = optim.Adam(linker.parameters(), lr=0.0005)
    
    mse_criterion = nn.MSELoss()    # 對齊錄音目標
    l1_criterion = nn.L1Loss()      # 【核心改進】保護 Model A 的原始清晰度
    
    early_stop = EarlyStopping(patience=40)

    # C. 分割資料集 (90% 訓練, 10% 驗證)
    full_dataset = FullSentenceDataset("metadata.txt")
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False, collate_fn=collate_fn)

    print("開始訓練 Model B (高保真卷積平滑器)...")
    
    epoch = 0
    while not early_stop.early_stop:
        epoch += 1
        linker.train()
        epoch_train_loss = 0
        
        for text_seq, mel_target in train_loader:
            optimizer.zero_grad()
            
            # 1. 透過 Model A 產生拼接片段
            with torch.no_grad():
                # 這裡會產生 [Batch, 字數 * 40, 80]
                chunks = [unit_gen(text_seq[:, i]) for i in range(text_seq.size(1))]
                # print('chunks length:', len(chunks))
                # print('each chunk shape:', chunks[0].shape)
                raw_mel = torch.cat(chunks, dim=1)
                
                # 線性插值對齊錄音長度
                raw_mel_aligned = F.interpolate(
                    raw_mel.transpose(1, 2), 
                    size=mel_target.size(1), 
                    mode='linear', 
                    align_corners=False
                ).transpose(1, 2)
            
            # 2. 透過 Model B 進行平滑
            # 訓練時使用 alpha=0.1 (稍大的干預讓模型學習，預測時再調小)
            output = linker(raw_mel_aligned)
            
            # 3. 計算組合損失
            # loss_mse: 負責銜接與語氣
            # loss_sim: 負責保住發音清晰度 (不能跟 A 差太多)
            loss_mse = mse_criterion(output, mel_target)
            loss_sim = l1_criterion(output, raw_mel_aligned)
            
            total_loss = loss_mse + 0.5 * loss_sim
            
            total_loss.backward()
            optimizer.step()
            epoch_train_loss += total_loss.item()

        # 4. 驗證階段
        linker.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            # print('text_seq shape:', text_seq.shape)
            for text_seq, mel_target in val_loader:
                chunks = [unit_gen(text_seq[:, i]) for i in range(text_seq.size(1))]
                raw_mel = torch.cat(chunks, dim=1)
                raw_mel_aligned = F.interpolate(raw_mel.transpose(1, 2), size=mel_target.size(1), mode='linear', align_corners=False).transpose(1, 2)
                
                output = linker(raw_mel_aligned)
                val_loss = mse_criterion(output, mel_target)
                epoch_val_loss += val_loss.item()
        
        avg_val_loss = epoch_val_loss / len(val_loader)
        improved = early_stop(avg_val_loss)
        
        if epoch % 5 == 0 or improved:
            msg = " (★ 最佳模型更新)" if improved else ""
            print(f"Epoch {epoch:3d}: Train Loss {epoch_train_loss/len(train_loader):.6f} | Val Loss {avg_val_loss:.6f}{msg}")
        
        if improved:
            torch.save(linker.state_dict(), "linker.pth")

    print(f"訓練自動終止。最佳驗證 Loss: {early_stop.best_loss:.6f}")

if __name__ == "__main__":
    train()