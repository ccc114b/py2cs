import os
from gtts import gTTS
from pypinyin import pinyin, Style

def gen_single_words():
    words = "你我他是愛有想的"  # 我們核心的 8 個字
    output_dir = "units_audio"
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    
    metadata = []
    for char in words:
        py = pinyin(char, style=Style.TONE3)[0][0]
        filename = f"{output_dir}/{py}.wav"
        # 產生單個字的音檔
        tts = gTTS(text=char, lang='zh-tw')
        tts.save(filename.replace(".wav", ".mp3"))
        # 建議手動轉 wav 或讓 librosa 處理 mp3
        metadata.append(f"{filename.replace('.wav', '.mp3')}|{py}")
        
    with open("metadata_units.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(metadata))
    print("單字音檔準備完成。")

if __name__ == "__main__":
    gen_single_words()
