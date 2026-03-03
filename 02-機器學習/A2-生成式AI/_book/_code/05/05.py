import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TransformerFoundations(nn.Module):
    def __init__(self, d_model=512, num_heads=8):
        super(TransformerFoundations, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

    def scaled_dot_product_attention(self, Q, K, V):
        """
        5.2 縮放點積注意力 (Scaled Dot-Product Attention) 的實作
        """
        # 計算點積相似度並進行縮放
        # $d_k$ 為鍵向量的維度
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(d_k)
        
        # 5.1 權重 $w_i$ 透過 Softmax 歸一化
        # $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$
        p_attn = F.softmax(scores, dim=-1)
        return torch.matmul(p_attn, V), p_attn

    def get_positional_encoding(self, seq_len, d_model):
        """
        5.3 位置編碼 (Positional Encoding) 的正餘弦實作
        """
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len).unsqueeze(1).float()
        # 計算 $10000^{2i/d_{model}}$
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))
        
        # 偶數維度使用 Sine，奇數維度使用 Cosine
        # $PE(pos, 2i) = \sin(pos/10000^{2i/d_{model}})$
        # $PE(pos, 2i+1) = \cos(pos/10000^{2i/d_{model}})$
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def multi_head_example(self, x):
        """
        5.4 多頭注意力 (Multi-Head Attention) 的簡化展示
        """
        batch_size, seq_len, d_model = x.size()
        
        # 假設已經進行了線性投影 $Q = XW^Q, K = XW^K, V = XW^V$
        # 這裡為了演示，直接將 d_model 拆分為多個頭
        query = x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        key = x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        value = x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # 針對每個頭進行並行運算
        x_out, attn_weights = self.scaled_dot_product_attention(query, key, value)
        
        # 拼接 (Concatenate) 各個頭
        x_out = x_out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return x_out, attn_weights

if __name__ == "__main__":
    # 初始化參數
    d_model = 64
    seq_len = 10
    batch_size = 1
    
    model_demo = TransformerFoundations(d_model=d_model, num_heads=4)
    
    # 模擬輸入序列向量
    input_vectors = torch.randn(batch_size, seq_len, d_model)
    
    print("--- 5.3 位置編碼測試 ---")
    pe = model_demo.get_positional_encoding(seq_len, d_model)
    print(f"位置編碼矩陣形狀: {pe.shape}")
    # 加上位置編碼
    input_with_pe = input_vectors + pe
    
    print("\n--- 5.2 & 5.4 縮放點積與多頭注意力測試 ---")
    output, weights = model_demo.multi_head_example(input_with_pe)
    
    print(f"輸入形狀: {input_vectors.shape}")
    print(f"輸出形狀: {output.shape} (應與輸入一致)")
    print(f"注意力權重形狀: {weights.shape} (Batch, Heads, Seq, Seq)")
    
    # 顯示第一個頭的前三個 token 的注意力分佈
    print("\n第一個頭的局部注意力權重 (前3x3):")
    print(weights[0, 0, :3, :3].detach().numpy())