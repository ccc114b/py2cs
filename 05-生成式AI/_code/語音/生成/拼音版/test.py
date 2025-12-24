import torch
import torchaudio
import json
import os
import soundfile as sf
import numpy as np
from pypinyin import pinyin, Style
from model import NeuralLinker
from model_unit import UnitGenerator

def tts_predict(text, alpha=0.02):
    # 1. 載入資源
    if not os.path.exists("char_map.json"):
        print("錯誤：找不到 char_map.json")
        return
    
    with open("char_map.json", "r", encoding="utf-8") as f:
        char_map = json.load(f)

    # 初始化模型
    unit_gen = UnitGenerator(vocab_size=len(char_map))
    linker = NeuralLinker()
    
    # 載入權重
    if not os.path.exists("unit_gen.pth") or not os.path.exists("linker.pth"):
        print("錯誤：找不到權重檔案 (.pth)")
        return
        
    unit_gen.load_state_dict(torch.load("unit_gen.pth", map_location='cpu'))
    linker.load_state_dict(torch.load("linker.pth", map_location='cpu'))
    unit_gen.eval()
    linker.eval()

    # 2. 文字轉拼音
    # pinyin 會回傳 [['wo3'], ['xiang3']...]
    py_list = [item[0] for item in pinyin(text, style=Style.TONE3)]
    print(f"拼音序列: {py_list}")
    
    # 轉換為 ID，並過濾掉不在字典裡的拼音
    tokens = [char_map[p] for p in py_list if p in char_map]
    
    # 【防錯檢查】如果 tokens 是空的，就停止執行
    if not tokens:
        print(f"錯誤：輸入的文字 '{text}' 中沒有包含可識別的字詞。")
        print(f"目前字典支援：{list(char_map.keys())}")
        return

    # 3. 推理
    with torch.no_grad():
        # A 產生 40 幀單元
        chunks = [unit_gen(torch.LongTensor([t])) for t in tokens]
        raw_mel = torch.cat(chunks, dim=1) # [1, Seq*40, 80]
        
        # B 卷積平滑 (傳入 alpha 參數)
        # alpha 越小，越接近 test_direct.py 的效果
        # alpha 越大，平滑感越強
        refined_mel = linker(raw_mel, alpha=alpha)
        
        # 4. 逆正規化 (必須對應訓練時的 log10 與 (x+2)/2)
        log_mel = (refined_mel * 2.0) - 2.0
        
        # --- 【核心改善：銳利化與去噪】 ---
        log_mel_np = log_mel.cpu().numpy()
        
        # 去噪門檻：將太小的數值直接壓死，消除背景嘶嘶聲
        # 門檻建議設定在最大值減去 2.5 到 3.0 之間
        gate_threshold = np.max(log_mel_np) - 2.8
        log_mel_np[log_mel_np < gate_threshold] = -5.0
        
        # 銳利化：增加對比度，讓聲音更清脆 (不喜歡可以改成 1.0)
        log_mel_np = log_mel_np * 1.05 
        
        log_mel = torch.from_numpy(log_mel_np)
        mel = torch.pow(10, log_mel).transpose(1, 2)

    # 5. 高品質還原 (Griffin-Lim)
    sample_rate = 22050
    griffin_lim = torchaudio.transforms.GriffinLim(
        n_fft=1024, hop_length=256, n_iter=150, momentum=0.99
    )
    inverse_mel = torchaudio.transforms.InverseMelScale(
        n_stft=513, n_mels=80, sample_rate=sample_rate
    )
    
    print(f"正在還原語音 (Alpha={alpha})...")
    waveform = griffin_lim(inverse_mel(mel)).squeeze().numpy()
    
    # 音量標準化
    if np.max(np.abs(waveform)) > 0:
        waveform = waveform / (np.max(np.abs(waveform)) + 1e-8)
    
    sf.write("final_output.wav", waveform, sample_rate)
    print("合成成功：final_output.wav")

if __name__ == "__main__":
    t = input("請輸入想要合成的句子: ").strip()
    
    # 如果使用者直接按 Enter，給予預設值
    if not t:
        t = "我想你是愛我的"
        print(f"使用預設文字: {t}")
        
    # 你可以嘗試調整 alpha 為 0.0, 0.02, 0.05
    # alpha=0.0 就會完全等於你的 test_direct.py
    tts_predict(t, alpha=0.01)