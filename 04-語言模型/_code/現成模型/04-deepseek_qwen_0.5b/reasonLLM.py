import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def run_test_cases():
    # 1. 設定模型名稱與裝置
    model_name = "Qwen/Qwen2.5-0.5B" # "deepseek-ai/DeepSeek-R1-Distill-Qwen-0.5B"
    
    # 偵測裝置 (MPS for Mac, CUDA for Nvidia, or CPU)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        dtype = torch.float16 # Mac 使用 float16 速度較快
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.bfloat16
    else:
        device = torch.device("cpu")
        dtype = torch.float32

    print(f"正在載入模型 {model_name} 到 {device}...")

    # 2. 載入分詞器與模型
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map={"": device} # 直接將模型對應到指定裝置
    )

    # 3. 定義測試案例
    test_cases = [
        "小明有 15 個蘋果，吃了 3 個，又買了 10 個，最後把一半送給小華，請問小明現在剩幾個？",
        "計算 25 * 4 + 125 / 5 等於多少？",
        "如果有 3 隻貓在 3 分鐘內可以抓到 3 隻老鼠，那麼 10 隻貓抓 10 隻老鼠需要幾分鐘？"
    ]

    print("\n" + "="*50)
    print("開始執行推理測試")
    print("="*50)

    for i, prob in enumerate(test_cases, 1):
        print(f"\n測試案例 {i}: {prob}")
        
        # 格式化 Prompt
        # R1 系列通常不需要特別指示，它看到問題就會開始 <thought>
        inputs = tokenizer(prob, return_tensors="pt").to(device)

        # 4. 生成答案
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=True,
                temperature=0.6, # 設定較低溫度以保持邏輯穩定
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print("-" * 20 + " 模型回覆 " + "-" * 20)
        print(response)
        print("=" * 50)

if __name__ == "__main__":
    run_test_cases()