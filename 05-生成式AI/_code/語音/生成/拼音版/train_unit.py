import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import librosa
import json
import os
import torch.nn.functional as F
from pypinyin import pinyin, Style
from model_unit import UnitGenerator

def train_units():
    # 1. 建立字元映射 (確保拼音與檔名一致)
    target_words = "你我他是愛有想的"
    word_to_py = {c: pinyin(c, style=Style.TONE3)[0][0] for c in target_words}
    pinyins = [word_to_py[c] for c in target_words]
    char_map = {py: i for i, py in enumerate(pinyins)}
    
    with open("char_map.json", "w", encoding="utf-8") as f:
        json.dump(char_map, f, ensure_ascii=False)

    # 2. 初始化模型 (確保對應 model_unit.py 的新結構)
    vocab_size = len(char_map)
    unit_gen = UnitGenerator(vocab_size=vocab_size)
    optimizer = optim.Adam(unit_gen.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # 梅爾頻譜轉換器
    mel_trans = torchaudio.transforms.MelSpectrogram(
        sample_rate=22050, 
        n_fft=1024, 
        hop_length=256, 
        n_mels=80,
        center=True
    )

    print(f"開始訓練 40 幀單字發音器 (Model A)... 詞彙量: {vocab_size}")
    
    # 增加訓練次數到 1500，確保 40 幀的高維度特徵能被完全背熟
    for epoch in range(1500):
        total_loss = 0
        for py, idx in char_map.items():
            optimizer.zero_grad()
            
            file_path = f"units_audio/{py}.mp3"
            if not os.path.exists(file_path): continue

            # 載入音訊
            y, _ = librosa.load(file_path, sr=22050)
            waveform = torch.from_numpy(y).unsqueeze(0)
            
            # 1. 轉梅爾頻譜 [1, 80, Time]
            target_mel = mel_trans(waveform)
            
            # 2. 強制對齊頻率維度為 80
            if target_mel.size(1) != 80:
                target_mel = F.interpolate(target_mel, size=(target_mel.size(2)), mode='bilinear', align_corners=False)
            
            # 3. 取 Log 與正規化 (必須與 predict/test 一致)
            target_mel = torch.log10(torch.clamp(target_mel, min=1e-5))
            target_mel = (target_mel + 2.0) / 2.0
            
            # 4. 【關鍵修改】強制縮放時間軸為 40 幀
            # 原本是 size=20，現在改為 size=40
            target_mel = F.interpolate(target_mel, size=40, mode='linear', align_corners=False)
            
            # 5. 轉置為 [40, 80] 以符合模型輸出 [Time, Mels]
            target_mel = target_mel.squeeze(0).transpose(0, 1) 
            
            # 6. 模型預測輸出為 [1, 40, 80] -> squeeze 變 [40, 80]
            output = unit_gen(torch.LongTensor([idx])).squeeze(0)
            
            # 7. 計算 Loss (此時兩者皆為 [40, 80])
            loss = criterion(output, target_mel)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1:4d}, Loss: {total_loss/len(char_map):.8f}")

    # 儲存權重
    torch.save(unit_gen.state_dict(), "unit_gen.pth")
    print("Model A (40幀版) 訓練完成！")

if __name__ == "__main__":
    train_units()