import torch
import torchaudio
import json
import soundfile as sf
from model_unit import UnitGenerator

def test_direct_concat(text):
    with open("char_map.json", "r") as f: char_map = json.load(f)
    unit_gen = UnitGenerator(len(char_map))
    unit_gen.load_state_dict(torch.load("unit_gen.pth", map_location='cpu'))
    unit_gen.eval()

    from pypinyin import pinyin, Style
    py_list = [item[0] for item in pinyin(text, style=Style.TONE3)]
    tokens = [char_map[p] for p in py_list if p in char_map]
    
    with torch.no_grad():
        chunks = [unit_gen(torch.LongTensor([t])) for t in tokens]
        raw_mel = torch.cat(chunks, dim=1)
        # 直接逆正規化
        log_mel = (raw_mel * 2.0) - 2.0
        mel = torch.pow(10, log_mel).transpose(1, 2)

    griffin_lim = torchaudio.transforms.GriffinLim(n_fft=1024, hop_length=256, n_iter=100)
    inverse_mel = torchaudio.transforms.InverseMelScale(n_stft=513, n_mels=80, sample_rate=22050)
    waveform = griffin_lim(inverse_mel(mel)).squeeze().numpy()
    sf.write("diagnostic_direct.wav", waveform, 22050)
    print("診斷檔案已產生：diagnostic_direct.wav (此檔案未經過 Model B)")

if __name__ == "__main__":
    test_direct_concat("我想你是愛我的")