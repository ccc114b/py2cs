import time

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

def run_test(name, text, window_size=100):
    print(f"=== 測試案例: {name} ===")
    lz = LZ77(window_size=window_size)
    
    start_time = time.time()
    compressed = lz.compress(text)
    end_time = time.time()
    
    decompressed = lz.decompress(compressed)
    
    # 統計資訊
    original_len = len(text)
    # 這裡簡單假設每個 token (offset, length, char) 佔用空間
    # 在真實儲存中，這些數字會被編碼成二進位，這裡僅計算 Tuple 數量當作指標
    num_tokens = len(compressed) 
    
    print(f"原始長度 : {original_len} 字元")
    print(f"壓縮後 Token 數 : {num_tokens} 個 Tuple")
    print(f"壓縮耗時 : {end_time - start_time:.4f} 秒")
    
    # 驗證
    is_valid = (text == decompressed)
    print(f"資料驗證 : {'成功 OK' if is_valid else '失敗 FAILED'}")
    
    if not is_valid:
        # 如果失敗，印出前50個字元比對
        print(f"預期: {text[:50]}...")
        print(f"實際: {decompressed[:50]}...")
    
    # 印出前 5 個壓縮結果讓你看結構
    print("前 5 個壓縮 Tuple:", compressed[:5])
    print("\n")

if __name__ == "__main__":
    # --- 範例 1: 高度重複的人工字串 ---
    # 模擬類似圖片底色或連續數據
    long_pattern = "A" * 50 + "B" * 50 + "AB" * 50 + ("LZ77_IS_COOL_" * 20)
    run_test("高度重複人工字串", long_pattern, window_size=255)

    # --- 範例 2: 經典英文童謠 (有大量重複單字) ---
    # The Wheels on the Bus
    lyrics = """
    The wheels on the bus go round and round,
    Round and round,
    Round and round.
    The wheels on the bus go round and round,
    All through the town.

    The wipers on the bus go swish, swish, swish,
    Swish, swish, swish,
    Swish, swish, swish.
    The wipers on the bus go swish, swish, swish,
    All through the town.
    """
    # 去除前後空白讓格式整齊
    lyrics = lyrics.strip()
    run_test("英文歌詞 (Wheels on the Bus)", lyrics, window_size=200)

    # --- 範例 3: 程式碼本身 (Meta Test) ---
    # 程式碼有大量的 self, def, return, print
    source_code_sample = """
    def compress(self, data):
        result = []
        cursor = 0
        while cursor < len(data):
            start = max(0, cursor - self.window_size)
            buffer = data[start:cursor]
            best_offset = 0
            best_length = 0
            # coding is fun coding is fun coding is fun
            if found:
                result.append((offset, length, char))
            else:
                result.append((0, 0, char))
        return result
    """ * 5  # 讓這段程式碼重複出現 5 次
    run_test("Python 程式碼 (重複 5 次)", source_code_sample, window_size=400)