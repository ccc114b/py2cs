def build_char_map(filename="sentences.txt"):
    # 1. 讀取檔案內容
    with open(filename, "r", encoding="utf-8") as f:
        text = f.read()

    # 2. 提取所有唯一字元，並排除換行符號
    # set() 會自動去重，sorted() 確保每次執行的索引順序一致
    unique_chars = sorted(list(set(text.replace('\n', ''))))

    # 3. 加入特殊標記：<PAD> 補長度, <UNK> 未知字
    # 建議將 <PAD> 放在 0，這樣在處理 batch padding 時比較方便
    special_tokens = ['<PAD>', '<UNK>']
    all_chars = special_tokens + unique_chars

    # 4. 建立映射表
    char_map = {c: i for i, c in enumerate(all_chars)}
    
    # 建立反向映射 (預測時會用到：數字轉文字)
    index_to_char = {i: c for c, i in char_map.items()}

    print(f"詞彙表建立完成，總共包含 {len(char_map)} 個字元。")
    return char_map, index_to_char


if __name__ == "__main__":
    # 在 train.py 中呼叫
    char_map, index_to_char = build_char_map("sentences.txt")

    # 測試看看
    test_text = "今天天氣不錯"
    tokens = [char_map.get(c, char_map['<UNK>']) for c in test_text]
    print(f"測試轉換: {test_text} -> {tokens}")
    import json

    # 儲存
    with open("char_map.json", "w", encoding="utf-8") as f:
        json.dump(char_map, f, ensure_ascii=False)
