import os
import time
from gtts import gTTS
from pypinyin import pinyin, Style

def to_pinyin(text):
    # 轉為帶聲調的拼音，例如: "你好" -> "ni3 hao3"
    py_list = pinyin(text, style=Style.TONE3)
    return " ".join([item[0] for item in py_list])

def generate_data(input_file="sentences.txt", output_folder="output_audio"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    metadata = []
    with open(input_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    print(f"開始處理 {len(lines)} 句語法...")
    for i, text in enumerate(lines, start=1):
        filename = f"{output_folder}/sentence_{i:04d}.mp3"
        py_text = to_pinyin(text)
        
        # 產生音檔 (若已存在則跳過)
        if not os.path.exists(filename):
            try:
                tts = gTTS(text=text, lang='zh-tw')
                tts.save(filename)
                print(f"[{i}/{len(lines)}] 已存: {filename}")
                time.sleep(0.8) # 避免被 Google 封鎖
            except Exception as e:
                print(f"跳過第 {i} 句，錯誤: {e}")
                continue
        
        metadata.append(f"{filename}|{py_text}")

    with open("metadata.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(metadata))
    print("--- 資料準備完成，metadata.txt 已產生 ---")

if __name__ == "__main__":
    generate_data()