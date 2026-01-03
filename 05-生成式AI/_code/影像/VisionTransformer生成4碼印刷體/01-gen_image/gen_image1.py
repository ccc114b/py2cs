import torch
import numpy as np
from PIL import Image, ImageDraw
import os

# --- 參數設定 ---
img_size = 64
output_dir = "test_samples"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 1. 複製之前的數據生成函數 (稍微微調以確保可測試)
def generate_fake_data(batch_size=10):
    texts = []
    images = []
    for _ in range(batch_size):
        # 隨機產生測試字串，包含數字、字母與標點
        age = np.random.randint(0, 100)
        types = [f"age={age}", f"val={age}!", f"ID:{age:02d}", f"T={age}C"]
        label_text = np.random.choice(types)
        
        # 建立黑底圖片
        img = Image.new('L', (img_size, img_size), color=0)
        d = ImageDraw.Draw(img)
        
        # 在 (5, 20) 的位置繪製文字
        # 註：這裡使用預設字體。在某些系統上可能很小。
        # 如果你有字體檔，可以改用 ImageFont.truetype("arial.ttf", 15)
        d.text((5, 20), label_text, fill=255)
        
        # 轉換為 numpy 陣列並歸一化 (0~1)
        img_array = np.array(img).astype(np.float32) / 255.0
        
        texts.append(label_text)
        images.append(img_array)
        
    return texts, torch.tensor(np.array(images))

# 2. 執行產生測試
print(f"正在產生 10 張範例圖片並存入 {output_dir}...")
texts, imgs = generate_fake_data(10)

# 建立一個畫布把 10 張圖拼在一起 (2x5 網格)
canvas_w = img_size * 5
canvas_h = img_size * 2
combined_img = Image.new('L', (canvas_w, canvas_h), color=128) # 灰色背景方便區分邊界

for i in range(10):
    # 將 Tensor 轉回 0-255 的圖片
    img_np = (imgs[i].numpy() * 255).astype(np.uint8)
    img_pil = Image.fromarray(img_np)
    
    # 計算在畫布上的位置
    x_offset = (i % 5) * img_size
    y_offset = (i // 5) * img_size
    combined_img.paste(img_pil, (x_offset, y_offset))
    
    # 分別儲存每一張，檔名包含文字內容
    # 注意：檔名不能含冒號等特殊字元，進行簡單處理
    safe_text = texts[i].replace(":", "_").replace("=", "-")
    img_pil.save(f"{output_dir}/sample_{i}_{safe_text}.png")
    
    print(f"圖片 {i}: 標籤為 '{texts[i]}'")

# 3. 儲存合併後的圖表
combined_img.save("all_samples_combined.png")
print("-" * 30)
print("測試完成！")
print(f"1. 個別圖片已儲存在 '{output_dir}' 資料夾中。")
print("2. 合併後的對照圖已儲存為 'all_samples_combined.png'。")

# 4. 如果你在有顯示介面的環境 (如 Local Python)，可以嘗試直接開啟
try:
    combined_img.show()
except:
    pass