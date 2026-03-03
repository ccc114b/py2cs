import torch
import torch.nn as nn
import torchaudio
import json
import os
import soundfile as sf
from pypinyin import pinyin, Style
from model import UnitGenerator, NeuralLinker

def predict(text, model_a_path="unit_gen.pth", model_b_path="linker.pth", char_map_path="char_map.json"):
    # 1. 載入字元映射表
    if not os.path.exists(char_map_path):
        print("錯誤：找不到 char_map.json，請先執行訓練。")
        return
    with open(char_map_path, "r", encoding="utf-8") as f:
        char_map = json.load(f)

    # 2. 初始化並載入模型權重
    unit_gen = UnitGenerator(vocab_size=len(char_map))
    linker = NeuralLinker()

    if not os.path.exists(model_a_path) or not os.path.exists(model_b_path):
        print("錯誤：找不到模型權重檔案 (.pth)")
        return

    unit_gen.load_state_dict(torch.load(model_a_path, map_location='cpu'))
    linker.load_state_dict(torch.load(model_b_path, map_location='cpu'))
    unit_gen.eval()
    linker.eval()

    # 3. 處理文字：中文 -> 拼音 -> ID 序列
    py_list = [item[0] for item in pinyin(text, style=Style.TONE3)]
    print(f"轉換拼音序列: {py_list}")
    
    tokens = [char_map.get(c, 1) for c in py_list] # 1 是 <UNK>
    
    # 4. 推理階段
    with torch.no_grad():
        # A. 模型 A 逐字產生頻譜單元 (每個字固定 20 幀)
        unit_mels = []
        for t in tokens:
            char_id = torch.LongTensor([t]) # [1]
            mel_chunk = unit_gen(char_id)    # 產生 [1, 20, 80]
            unit_mels.append(mel_chunk)
        
        # 拼接所有單元
        raw_concat_mel = torch.cat(unit_mels, dim=1) # [1, Seq_Len*20, 80]
        
        # B. 模型 B 進行平滑處理與韻律調整
        # Transformer 會觀察前後文，讓拼接處不那麼突兀
        refined_mel = linker(raw_concat_mel) # [1, T, 80]

        # 5. 逆正規化與轉換 (假設訓練時使用了 log10 正規化)
        # 如果訓練時是用 (log_mel + 2.0) / 2.0，這裡要還原
        # log_mel = (refined_mel * 2.0) - 2.0
        log_mel = refined_mel 
        
        mel_output = torch.pow(10, log_mel).transpose(1, 2) # [1, 80, T]

    # 6. Griffin-Lim 演算法還原音訊
    sample_rate = 22050
    n_fft = 1024
    hop_length = 256

    inverse_mel_scale = torchaudio.transforms.InverseMelScale(
        n_stft=n_fft // 2 + 1, n_mels=80, sample_rate=sample_rate
    )
    griffin_lim = torchaudio.transforms.GriffinLim(
        n_fft=n_fft, hop_length=hop_length, win_length=n_fft
    )

    print("正在還原音訊波形...")
    spec = inverse_mel_scale(mel_output)
    waveform = griffin_lim(spec)
    
    # 7. 儲存音檔
    audio_data = waveform.squeeze().numpy()
    output_file = "neural_output.wav"
    sf.write(output_file, audio_data, sample_rate)
    
    print(f"成功！音檔已儲存：{output_file}")

if __name__ == "__main__":
    # 限制詞彙測試
    test_text = input("請輸入想要合成的句子 (建議使用: 你, 我, 他, 是, 有, 愛, 想, 的): ")
    if not test_text:
        test_text = "我想你是愛我的"
    
    predict(test_text)