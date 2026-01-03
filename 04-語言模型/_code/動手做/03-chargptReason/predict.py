import torch
from model import CharGPT, get_device

def main():
    device = get_device()
    
    if not torch.os.path.exists('chargpt_math.pth'):
        print("找不到模型文件！")
        return

    # 載入 Checkpoint
    ckpt = torch.load('chargpt_math.pth', map_location=device)
    stoi, itos = ckpt['stoi'], ckpt['itos']
    encode = lambda s: [stoi[c] for c in s if c in stoi]
    decode = lambda l: ''.join([itos[i] for i in l])

    # 根據儲存的設定重建模型
    model = CharGPT(**ckpt['config'], device=device)
    model.load_state_dict(ckpt['model_state'])
    model.to(device)
    model.eval()

    print("模型載入成功！現在可以輸入數學題了。")
    
    while True:
        q = input("\n問題 (或輸入 exit): ")
        if q.lower() == 'exit': break
        
        prompt = f"Q: {q}\n<thought>\n"
        x = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
        
        # 生成答案
        with torch.no_grad():
            y = model.generate(x, max_new_tokens=150)
            output = decode(y[0].tolist())
            
        # 只印出推導過程與答案
        print("\nAI 推理過程與結果：")
        print(output[len(f"Q: {q}\n"):])

if __name__ == "__main__":
    main()