import torch
import sys
from model import CharGPT, get_device

device = get_device()
SAVE_PATH = "gpt_model.pt"

def run_zen_examples():
    try:
        # 載入權重與配置
        checkpoint = torch.load(SAVE_PATH, map_location=device)
    except FileNotFoundError:
        print("錯誤：找不到模型存檔 'gpt_model.pt'，請先執行訓練程式。")
        return

    config = checkpoint['config']
    itos = checkpoint['itos']
    stoi = checkpoint['stoi']
    vocab_size = checkpoint['vocab_size']

    # 初始化模型
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
    model.eval()

    encode = lambda s: [stoi[c] for c in s if c in stoi]
    decode = lambda l: ''.join([itos[i] for i in l])

    # 預設的十個範例問句
    example_questions = [
        "什麼是道？",
        "如何才能開悟？",
        "心在哪裡？",
        "佛在哪裡？",
        "什麼是生死？",
        "如何放下執著？",
        "我是誰？",
        "世界是真的嗎？",
        "為何人生這麼苦？",
        "如何得到真正的自由？"
    ]

    print(f"--- 禪宗大師 GPT 範例展示 (設備: {device}) ---\n")

    for i, q in enumerate(example_questions, 1):
        # 構造 Prompt，模擬訓練時的格式
        prompt = f"問：{q} 答："
        context_ids = encode(prompt)
        context_tensor = torch.tensor([context_ids], dtype=torch.long, device=device)
        
        # 生成答案
        # 注意：max_new_tokens 設為 100 足以產生一句完整的回答
        generated_idx = model.generate(context_tensor, max_new_tokens=100)
        
        # 取得生成的部分（扣除 prompt 本身長度）
        new_tokens = generated_idx[0, len(context_ids):].tolist()
        response = decode(new_tokens)
        
        # 擷取答案：取第一個「問：」之前的內容，確保不會抓到模型自問自答的下一題
        answer = response.split("問：")[0].strip()
        
        print(f"範例 {i}")
        print(f"問：{q}")
        print(f"答：{answer}")
        print("-" * 20)

if __name__ == "__main__":
    run_zen_examples()