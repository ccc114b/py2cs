import torch
import torchaudio
import json
import soundfile as sf
import os
import numpy as np
from pypinyin import pinyin, Style
from model_transformer import NeuralLinker
from model_unit import UnitGenerator

def tts_predict(text):
    with open("char_map.json", "r", encoding="utf-8") as f:
        char_map = json.load(f)

    unit_gen = UnitGenerator(len(char_map))
    linker = NeuralLinker()
    unit_gen.load_state_dict(torch.load("unit_gen.pth", map_location='cpu'))
    linker.load_state_dict(torch.load("linker_transformer.pth", map_location='cpu'))
    unit_gen.eval()
    linker.eval()

    py_list = [item[0] for item in pinyin(text, style=Style.TONE3)]
    tokens = [char_map[p] for p in py_list if p in char_map]
    
    if not tokens: return

    print(f"拼音序列: {py_list}，正在推理...")
    with torch.no_grad():
        chunks = [unit_gen(torch.LongTensor([t])) for t in tokens]
        raw_mel = torch.cat(chunks, dim=1)
        # B 模型平滑修正
        refined_mel = linker(raw_mel)
        # 逆正規化
        log_mel = (refined_mel * 2.0) - 2.0
        mel = torch.pow(10, log_mel).transpose(1, 2)

    # Griffin-Lim 還原 (迭代 100 次以獲得清晰音質)
    griffin_lim = torchaudio.transforms.GriffinLim(n_fft=1024, hop_length=256, n_iter=100)
    inverse_mel = torchaudio.transforms.InverseMelScale(n_stft=513, n_mels=80, sample_rate=22050)
    
    waveform = griffin_lim(inverse_mel(mel)).squeeze().numpy()
    
    # 音量標準化
    if np.max(np.abs(waveform)) > 0:
        waveform = waveform / np.max(np.abs(waveform))
    
    sf.write("final_output_transformer.wav", waveform, 22050)
    print("合成成功：final_output_transformer.wav")

if __name__ == "__main__":
    t = input("請輸入想要合成的句子: ")
    tts_predict(t if t else "我想你是愛我的")