import random
import json

def generate_math_reasoning_data(num_samples=30000):
    dataset = []
    
    for i in range(num_samples):
        # 隨機選擇題型
        type_choice = random.choice(['pure_calc', 'word_problem', 'comparison'])
        
        if type_choice == 'pure_calc':
            # 題型 1: 純算式 (a + b - c)
            a = random.randint(1, 20)
            b = random.randint(1, 20)
            c = random.randint(1, a + b) # 確保結果不為負數
            
            question = f"{a} + {b} - {c} = ?"
            thought = f"首先，計算前面的加法：{a} + {b} 等於 {a+b}。接著，從 {a+b} 中減去 {c}。最終計算結果是 {a+b-c}。"
            answer = str(a + b - c)

        elif type_choice == 'word_problem':
            # 題型 2: 簡單應用題
            names = ["小明", "小紅", "小華", "小強"]
            items = ["蘋果", "糖果", "筆", "球"]
            name = random.choice(names)
            item = random.choice(items)
            
            a = random.randint(5, 15)
            b = random.randint(1, 10)
            c = random.randint(1, a)
            
            question = f"{name}原本有 {a} 個{item}，買了 {b} 個後，送給朋友 {c} 個，請問{name}現在還有幾個{item}？"
            thought = f"1. 初始數量是 {a}。2. 買了 {b} 個，所以數量增加：{a} + {b} = {a+b}。3. 送給朋友 {c} 個，所以數量減少：{a+b} - {c} = {a+b-c}。"
            answer = str(a + b - c)

        else:
            # 題型 3: 比較大小
            a, b = random.randint(1, 10), random.randint(1, 10)
            c, d = random.randint(1, 10), random.randint(1, 10)
            
            question = f"請問 {a}+{b} 是否大於 {c}+{d}？"
            sum1 = a + b
            sum2 = c + d
            result = "是" if sum1 > sum2 else "否"
            
            thought = f"左邊的結果是 {a} + {b} = {sum1}。右邊的結果是 {c} + {d} = {sum2}。比較 {sum1} 和 {sum2}，答案為{result}。"
            answer = result

        # 格式化成模型訓練用的樣子
        # 我們使用標籤 <thought> 來包裹推理過程
        full_text = f"Q: {question}\n<thought>\n{thought}\n</thought>\nA: {answer}"
        
        dataset.append({"text": full_text})

    # 儲存為 JSONL 格式
    with open('math_dataset.jsonl', 'w', encoding='utf-8') as f:
        for entry in dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"成功生成 {num_samples} 條數據，儲存在 math_dataset.jsonl")

if __name__ == "__main__":
    generate_math_reasoning_data()