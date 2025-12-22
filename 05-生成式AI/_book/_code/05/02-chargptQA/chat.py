import torch
import sys
from model import CharGPT

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
SAVE_PATH = "gpt_model.pt"

def load_and_chat():
    try:
        checkpoint = torch.load(SAVE_PATH, map_location=device)
    except FileNotFoundError:
        print("錯誤：找不到模型存檔，請先執行 train.py")
        return

    # 從存檔中還原配置
    config = checkpoint['config']
    itos = checkpoint['itos']
    stoi = checkpoint['stoi']
    vocab_size = checkpoint['vocab_size']

    # 初始化模型架構並載入權重
    model = CharGPT(
        vocab_size=vocab_size,
        n_embd=config['n_embd'],
        n_head=config['n_head'],
        n_layer=config['n_layer'],
        block_size=config['block_size'],
        dropout=config['dropout'],
        device=device
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval() # 切換到評估模式

    encode = lambda s: [stoi[c] for c in s if c in stoi]
    decode = lambda l: ''.join([itos[i] for i in l])

    print("-" * 30)
    print("GPT 載入成功！輸入內容後按 Enter 讓模型續寫。")
    print("輸入 'quit' 或 'exit' 結束。")
    print("-" * 30)

    while True:
        user_input = input("\n問：")
        if user_input.lower() in ['quit', 'exit']:
            break
        
        if not user_input:
            continue

        # 將輸入轉換為 Tensor
        context = torch.tensor([encode("問："+user_input)], dtype=torch.long, device=device)
        
        # 生成續寫內容 (例如生成 100 個字元)
        print("答：", end="")
        generated_idx = model.generate(context, max_new_tokens=100)
        # 只顯示續寫的部分 (扣除原輸入)
        new_tokens = generated_idx[0, len(user_input):].tolist()
        response = decode(new_tokens)
        answer = response.split("答：")[1].split("問：")[0].strip()
        print(answer)

if __name__ == "__main__":
    load_and_chat()