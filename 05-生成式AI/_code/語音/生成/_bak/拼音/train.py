import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import librosa
import json
import os
import numpy as np
from torch.utils.data import DataLoader, Dataset
from model import UnitGenerator, NeuralLinker

# 1. 建立字元映射表
def build_char_map(metadata_path="metadata.txt"):
    all_text = ""
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            if '|' in line:
                all_text += line.split('|')[1]
    
    # 1. 取得所有唯一的字元
    chars_in_text = set(all_text.replace('\n', ''))
    
    # 2. 定義必須存在的特殊字元
    special_tokens = ['<PAD>', '<UNK>', ' ']
    
    # 3. 從文本字元中移除已經在特殊字元裡的符號，避免重複
    remaining_chars = sorted(list(chars_in_text - set(special_tokens)))
    
    # 4. 合併最終清單
    all_chars = special_tokens + remaining_chars
    
    # 5. 建立映射
    char_map = {c: i for i, c in enumerate(all_chars)}
    
    with open("char_map.json", "w", encoding="utf-8") as f:
        json.dump(char_map, f, ensure_ascii=False)
        
    print(f"建立詞彙表成功，總數: {len(char_map)}")
    return char_map

# 2. 資料集讀取與音訊預處理
class TTSDataset(Dataset):
    def __init__(self, metadata_path, char_map):
        self.char_map = char_map
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.data = [line.strip().split('|') for line in f if '|' in line]
        
        # 梅爾頻譜轉換器
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=22050, 
            n_fft=1024, 
            hop_length=256, 
            n_mels=80
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_path, py_text = self.data[idx]
        
        # 使用 librosa 載入音訊 (對 M1 Mac 較穩定)
        y, sr = librosa.load(audio_path, sr=22050)
        waveform = torch.from_numpy(y).unsqueeze(0)
        
        # 轉梅爾頻譜並取 Log
        mel_spec = self.mel_transform(waveform)
        # 數值正規化，讓模型好練
        log_mel = torch.log10(torch.clamp(mel_spec, min=1e-5))
        log_mel = (log_mel + 2.0) / 2.0 # 縮放到約 0~1 區間
        
        # 文字轉 ID 序列
        tokens = [self.char_map.get(c, 1) for c in py_text]
        
        return torch.LongTensor(tokens), log_mel.squeeze(0).T

# 3. Batch 處理 (補零對齊)
def collate_fn(batch):
    texts, mels = zip(*batch)
    texts_pad = torch.nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=0)
    
    max_mel_len = max(m.size(0) for m in mels)
    mels_pad = torch.zeros(len(mels), max_mel_len, 80)
    for i, m in enumerate(mels):
        mels_pad[i, :m.size(0), :] = m
        
    return texts_pad, mels_pad

# 4. 訓練主程式
def train():
    # A. 準備資料
    if not os.path.exists("metadata.txt"):
        print("錯誤：找不到 metadata.txt，請先執行 gen.py")
        return

    char_map = build_char_map()
    dataset = TTSDataset("metadata.txt", char_map)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

    # B. 初始化雙模型
    vocab_size = len(char_map)
    unit_gen = UnitGenerator(vocab_size=vocab_size)
    linker = NeuralLinker()

    optimizer = optim.Adam(
        list(unit_gen.parameters()) + list(linker.parameters()), 
        lr=0.001
    )
    criterion = nn.MSELoss()

    print(f"開始雙模型訓練... 詞彙量: {vocab_size}")

    # C. 訓練迴圈
    num_epochs = 200 # 資料量少，建議跑 200 次以上
    for epoch in range(num_epochs):
        total_loss = 0
        for text_seq, mel_target in dataloader:
            optimizer.zero_grad()
            
            # --- 第一階段：Model A 產生各個字元的發音單元 ---
            batch_size = text_seq.size(0)
            seq_len = text_seq.size(1)
            
            # 收集每個字元的單元 [B, 20, 80]
            unit_chunks = []
            for i in range(seq_len):
                char_ids = text_seq[:, i] # 取得 Batch 中第 i 個字
                chunk = unit_gen(char_ids) 
                unit_chunks.append(chunk)
            
            # 拼接成原始頻譜 [B, Seq_Len * 20, 80]
            raw_concat_mel = torch.cat(unit_chunks, dim=1)
            
            # --- 第二階段：長度對齊與 Model B 平滑 ---
            # 將 Model A 的輸出長度強制縮放至與目標錄音檔一致
            target_len = mel_target.size(1)
            raw_concat_mel = raw_concat_mel.transpose(1, 2) # [B, 80, S*20]
            raw_concat_mel = torch.nn.functional.interpolate(
                raw_concat_mel, size=target_len, mode='linear', align_corners=False
            )
            raw_concat_mel = raw_concat_mel.transpose(1, 2) # [B, target_len, 80]
            
            # 透過 Model B (Transformer) 優化連接處
            refined_mel = linker(raw_concat_mel)
            
            # --- 計算損失與更新 ---
            loss = criterion(refined_mel, mel_target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.6f}")

    # D. 儲存權重
    torch.save(unit_gen.state_dict(), "unit_gen.pth")
    torch.save(linker.state_dict(), "linker.pth")
    print("訓練完成！已產生 unit_gen.pth 與 linker.pth")

if __name__ == "__main__":
    train()