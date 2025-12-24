import os
import time
from gtts import gTTS

def text_to_speech_batch(input_file, output_folder="output_audio", metadata_file="metadata.txt"):
    # 1. 建立輸出資料夾
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"建立資料夾: {output_folder}")

    # 2. 讀取所有語句
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"錯誤：找不到檔案 {input_file}")
        return

    total = len(lines)
    print(f"開始轉換，預計總數：{total}")

    # 用來儲存 metadata 的內容
    metadata_records = []

    # 3. 逐行轉換
    for index, text in enumerate(lines, start=1):
        filename = f"{output_folder}/sentence_{index:04d}.mp3"
        
        # 檢查是否已經存在（方便中斷後續傳）
        if os.path.exists(filename):
            print(f"[{index}/{total}] 跳過已存在的檔案: {filename}")
            metadata_records.append(f"{filename}|{text}")
            continue

        try:
            # 產生語音
            tts = gTTS(text=text, lang='zh-tw', slow=False)
            tts.save(filename)
            print(f"[{index}/{total}] 已完成: {filename}")
            
            # 加入 metadata 紀錄
            metadata_records.append(f"{filename}|{text}")
            
            # 重要：加入短暫延遲防止被 Google 封鎖
            time.sleep(0.8) 

        except Exception as e:
            print(f"[{index}/{total}] 發生錯誤: {text}. 錯誤原因: {e}")
            # 如果發生錯誤，就不加入 metadata，避免訓練時找不到檔案
            time.sleep(5)

    # 4. 產生 metadata.txt
    print(f"\n正在寫入 {metadata_file}...")
    with open(metadata_file, "w", encoding="utf-8") as f:
        for line in metadata_records:
            f.write(line + "\n")

    print(f"--- 所有任務處理完畢，共紀錄 {len(metadata_records)} 筆資料 ---")

if __name__ == "__main__":
    # 執行轉換
    text_to_speech_batch("sentences.txt")