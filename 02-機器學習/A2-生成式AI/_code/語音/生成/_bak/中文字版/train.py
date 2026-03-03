import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import librosa
import json
import os
from torch.utils.data import DataLoader, Dataset
from model import SimpleTTS

epochs = 500

def build_char_map(filename="sentences.txt"):
    with open(filename, "r", encoding="utf-8") as f:
        text = f.read()
    unique_chars = sorted(list(set(text.replace('\n', ''))))
    special_tokens = ['<PAD>', '<UNK>']
    all_chars = special_tokens + unique_chars
    char_map = {c: i for i, c in enumerate(all_chars)}
    with open("char_map.json", "w", encoding="utf-8") as f:
        json.dump(char_map, f, ensure_ascii=False)
    return char_map

class TTSDataset(Dataset):
    def __init__(self, metadata_path, char_map):
        self.char_map = char_map
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.data = [line.strip().split('|') for line in f if '|' in line]
        
        # 調整梅爾頻譜參數以消除警告
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=22050, 
            n_fft=1024,     # 增加 FFT 視窗大小
            win_length=1024, 
            hop_length=256, 
            n_mels=80
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_path, text = self.data[idx]
        y, sr = librosa.load(audio_path, sr=22050) 
        waveform = torch.from_numpy(y).unsqueeze(0)
        
        # 轉梅爾頻譜並取 Log (TTS 常用 log-mel)
        mel_spec = self.mel_transform(waveform)
        mel_spec = torch.log10(mel_spec + 1e-9) 
        
        tokens = [self.char_map.get(c, self.char_map['<UNK>']) for c in text]
        return torch.LongTensor(tokens), mel_spec.squeeze(0).T

def collate_fn(batch):
    texts, mels = zip(*batch)
    texts_pad = torch.nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=0)
    
    # 這裡我們將所有頻譜圖補齊到該 Batch 的最大長度
    max_mel_len = max(m.size(0) for m in mels)
    mels_pad = torch.zeros(len(mels), max_mel_len, 80)
    for i, m in enumerate(mels):
        mels_pad[i, :m.size(0), :] = m
        
    return texts_pad, mels_pad

def train():
    char_map = build_char_map("sentences.txt")
    vocab_size = len(char_map)
    dataset = TTSDataset("metadata.txt", char_map)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

    model = SimpleTTS(vocab_size=vocab_size)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    print(f"開始訓練... 詞彙量: {vocab_size}")
    for epoch in range(epochs):
        total_loss = 0
        for text_seq, mel_target in dataloader:
            optimizer.zero_grad()
            
            # 關鍵點：傳入目標長度 (mel_target.size(1))
            output = model(text_seq, target_len=mel_target.size(1))
            
            loss = criterion(output, mel_target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")

    torch.save(model.state_dict(), "tts_model.pth")
    print("訓練完成！")

if __name__ == "__main__":
    train()