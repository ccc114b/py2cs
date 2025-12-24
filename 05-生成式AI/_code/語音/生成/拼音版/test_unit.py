import torch
import torchaudio
import json
import os
import soundfile as sf
import numpy as np
from model_unit import UnitGenerator

def test_single_unit(target_pinyin="wo3", model_path="unit_gen.pth", char_map_path="char_map.json"):
    # 1. 載入字元映射表
    if not os.path.exists(char_map_path):
        print(f"錯誤：找不到 {char_map_path}")
        return
    with open(char_map_path, "r", encoding="utf-8") as f:
        char_map = json.load(f)

    if target_pinyin not in char_map:
        print(f"錯誤：拼音 [{target_pinyin}] 不在字典中。目前的字典有：{list(char_map.keys())}")
        return

    # 2. 載入模型 (結構會自動讀取 model_unit.py 中的 40 幀設定)
    vocab_size = len(char_map)
    unit_gen = UnitGenerator(vocab_size=vocab_size)
    
    if not os.path.exists(model_path):
        print(f"錯誤：找不到模型權重 {model_path}")
        return
    
    unit_gen.load_state_dict(torch.load(model_path, map_location='cpu'))
    unit_gen.eval()

    # 3. 推理：將拼音轉為頻譜
    idx = char_map[target_pinyin]
    char_id = torch.LongTensor([idx])
    
    print(f"正在產生拼音 [{target_pinyin}] 的聲音 (40 幀解析度)...")
    with torch.no_grad():
        # 模型輸出現在是: [1, 40, 80]
        refined_mel = unit_gen(char_id)
        
        # 逆正規化：對應訓練時的 (log_mel + 2.0) / 2.0
        log_mel = (refined_mel * 2.0) - 2.0
        
        # 轉回梅爾頻譜：從 [1, 40, 80] 轉為 [1, 80, 40]
        mel_output = torch.pow(10, log_mel).transpose(1, 2)

    # 4. Griffin-Lim 還原音訊設定
    sample_rate = 22050
    n_fft = 1024
    hop_length = 256

    inverse_mel_scale = torchaudio.transforms.InverseMelScale(
        n_stft=n_fft // 2 + 1, n_mels=80, sample_rate=sample_rate
    )
    
    # 【優化】增加 n_iter 到 100，讓 40 幀的相位收斂更完整，減少雜訊
    griffin_lim = torchaudio.transforms.GriffinLim(
        n_fft=n_fft, 
        hop_length=hop_length, 
        win_length=n_fft,
        n_iter=100 
    )

    print("正在將頻譜還原為波形...")
    spec = inverse_mel_scale(mel_output)
    waveform = griffin_lim(spec)
    
    # 5. 儲存結果與音量標準化
    audio_data = waveform.squeeze().numpy()
    
    # 簡單的音量標準化，防止聲音太小
    if np.max(np.abs(audio_data)) > 0:
        audio_data = audio_data / np.max(np.abs(audio_data))
        
    output_filename = f"test_{target_pinyin}.wav"
    sf.write(output_filename, audio_data, sample_rate)
    
    print(f"成功！請聽聽看檔案：{output_filename}")

if __name__ == "__main__":
    test_pinyin = input("請輸入想測試的拼音 (例如 wo3): ")
    if not test_pinyin:
        test_pinyin = "wo3"
        
    test_single_unit(test_pinyin)