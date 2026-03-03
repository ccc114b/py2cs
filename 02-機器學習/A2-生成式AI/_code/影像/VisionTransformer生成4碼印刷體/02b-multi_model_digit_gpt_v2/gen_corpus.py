import random

def create_dataset(filename="dataset.txt", count=1000):
    verbs = ["畫出", "寫出", "產生", "顯示", "繪製", "呈現", "寫", "畫", "給"]
    polite = ["請", "幫我", "麻煩", "", "立刻", "可否"]
    connectors = [" ", "數字", "個", ":", "如下"]
    
    with open(filename, "w", encoding="utf-8") as f:
        for _ in range(count):
            v, p, c = random.choice(verbs), random.choice(polite), random.choice(connectors)
            num_str = str(random.randint(0, 9999))
            
            # 組合句子
            sentence = f"{p}{v}{c}{num_str}".strip().replace("  ", " ")
            # 儲存格式：句子|數字 (方便訓練時提取數字繪圖)
            f.write(f"{sentence}|{num_str}\n")
    print(f"成功產生 {count} 句指令並存入 {filename}")

if __name__ == "__main__":
    create_dataset()