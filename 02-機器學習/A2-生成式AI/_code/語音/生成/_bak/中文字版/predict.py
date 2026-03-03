import torch
import torchaudio
import json
import os
import soundfile as sf  # 引入 soundfile
from model import SimpleTTS

def predict(text, model_path="tts_model.pth", char_map_path="char_map.json", output_file="output.wav"):
    # 1. 載入字元映射表
    if not os.path.exists(char_map_path):
        print(f"錯誤：找不到 {char_map_path}，請先執行 train.py")
        return
    with open(char_map_path, "r", encoding="utf-8") as f:
        char_map = json.load(f)

    vocab_size = len(char_map)

    # 2. 載入模型
    model = SimpleTTS(vocab_size=vocab_size)
    if not os.path.exists(model_path):
        print(f"錯誤：找不到模型檔案 {model_path}")
        return
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # 3. 處理輸入文字
    tokens = [char_map.get(c, char_map.get('<UNK>', 1)) for c in text]
    text_tensor = torch.LongTensor([tokens])

    # 4. 估計音訊長度 (1個字對應15幀)
    frames_per_char = 15
    target_len = len(tokens) * frames_per_char

    # 5. 推理 (Inference)
    print(f"正在生成語音，預計長度：{target_len} 幀...")
    with torch.no_grad():
        log_mel_output = model(text_tensor, target_len=target_len)
        mel_output = torch.pow(10, log_mel_output) - 1e-9
        mel_output = mel_output.transpose(1, 2) # [1, 80, T]

    # 6. 將梅爾頻譜轉回波形
    sample_rate = 22050
    n_fft = 1024
    hop_length = 256
    win_length = 1024

    inverse_mel_scale = torchaudio.transforms.InverseMelScale(
        n_stft=n_fft // 2 + 1, n_mels=80, sample_rate=sample_rate
    )
    griffin_lim = torchaudio.transforms.GriffinLim(
        n_fft=n_fft, hop_length=hop_length, win_length=win_length
    )

    print("正在轉換頻譜圖為音訊...")
    spec = inverse_mel_scale(mel_output)
    waveform = griffin_lim(spec) # [1, T]

    # 7. 使用 soundfile 儲存音檔 (避開 torchaudio.save 的 torchcodec 依賴)
    # waveform.squeeze() 將 [1, T] 轉為 [T]
    audio_data = waveform.squeeze().numpy()
    
    # 儲存為 wav 檔
    sf.write(output_file, audio_data, sample_rate)
    
    print(f"成功！音檔已儲存至：{output_file}")

if __name__ == "__main__":
    input_text = input("請輸入想要轉換的中文句子：")
    if not input_text:
        input_text = "我想他"
    
    predict(input_text)