class LZ77:
    def __init__(self, window_size=20):
        self.window_size = window_size

    def compress(self, data):
        """
        壓縮字串
        回傳格式: List of tuples (offset, length, next_char)
        """
        result = []
        cursor = 0  # 目前處理到的位置
        data_len = len(data)

        while cursor < data_len:
            # 1. 定義搜尋視窗 (Search Buffer) - 已經看過的資料
            start_index = max(0, cursor - self.window_size)
            search_buffer = data[start_index:cursor]
            
            # 2. 定義前方預覽視窗 (Lookahead Buffer) - 準備要壓縮的資料
            # 這裡簡單設定預覽長度跟視窗一樣，或者直到字串結束
            lookahead_buffer = data[cursor:min(cursor + self.window_size, data_len)]
            
            best_offset = 0
            best_length = 0
            best_char = ''

            # 3. 在搜尋視窗中尋找最長的匹配字串
            # 我們嘗試從預覽視窗拿出越來越長的字串，看是否曾在搜尋視窗出現過
            found_match = False
            
            # 嘗試長度從 1 到 lookahead 的長度
            for length in range(1, len(lookahead_buffer) + 1):
                substring = lookahead_buffer[:length]
                # 在 search_buffer 找最後一次出現的位置 (通常離 cursor 越近越好，或視實作而定)
                index_in_buffer = search_buffer.rfind(substring)
                
                if index_in_buffer != -1:
                    # 找到了！記錄下來，並嘗試找更長的
                    found_match = True
                    best_length = length
                    # offset 是「往前數幾個字元」
                    # search_buffer 長度 - index 就是往回的距離
                    best_offset = len(search_buffer) - index_in_buffer
                else:
                    # 如果目前的長度找不到，更長的也不會找到，直接跳出
                    break
            
            # 4. 決定下一個字元 (next_char)
            if cursor + best_length < data_len:
                best_char = data[cursor + best_length]
            else:
                best_char = '' # 已經到結尾

            # 5. 寫入結果並移動游標
            # 輸出的 tuple: (往前幾格, 複製多長, 下一個新字元)
            result.append((best_offset, best_length, best_char))
            
            # 游標移動量 = 匹配長度 + 1 (那個 next_char)
            cursor += best_length + 1

        return result

    def decompress(self, compressed_data):
        """
        解壓縮
        """
        output = []
        
        for offset, length, char in compressed_data:
            if length == 0:
                # 沒有匹配，直接加入字元
                if char: output.append(char)
            else:
                # 有匹配，回頭找資料複製
                # 目前 output 的長度
                current_len = len(output)
                # 起始點
                start = current_len - offset
                
                # 簡單的複製邏輯 (注意：LZ77 允許匹配長度超過 offset，這裡用簡單切片處理)
                # 為了處理 length > offset 的情況（例如 "aaaaa"），我們一個個字元複製
                for i in range(length):
                    output.append(output[start + i])
                
                # 最後加上 next_char
                if char:
                    output.append(char)
                    
        return "".join(output)

# --- 測試程式碼 ---

if __name__ == "__main__":
    # 測試字串：有明顯重複模式
    text = "CABRACADABRA" 
    
    lz = LZ77(window_size=10)
    
    # 1. 壓縮
    compressed = lz.compress(text)
    print(f"原始字串: {text}")
    print(f"原始長度: {len(text)}")
    print("-" * 30)
    print("壓縮結果 (Offset, Length, Next_Char):")
    for item in compressed:
        print(item)
    
    # 2. 解壓縮
    decompressed = lz.decompress(compressed)
    print("-" * 30)
    print(f"解壓結果: {decompressed}")
    
    # 3. 驗證
    assert text == decompressed
    print(f"驗證成功: {text == decompressed}")
