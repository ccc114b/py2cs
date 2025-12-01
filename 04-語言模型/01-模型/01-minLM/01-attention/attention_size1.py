import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- 1. Scaled Dot-Product Attention (縮放點積注意力) 函式 ---
# 這是 Attention 機制的核心計算部分：Q K^T / sqrt(d_k)
def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    計算 Attention(Q, K, V) = Softmax(Q K^T / sqrt(d_k)) * V

    Args:
        Q, K, V: 查詢、鍵、值矩陣 (Tensor)。
        mask: 遮罩矩陣 (用於避免看到未來 Token，GPT 專用)。
    """
    # 獲取 Q, K, V 的維度 (通常是 batch_size, num_heads, seq_len, head_dim)
    # d_k: K 向量的維度 (head_dim)
    d_k = Q.size(-1)

    # 1. 計算相似度分數 (QK^T)
    # (..., seq_len_q, d_k) @ (..., d_k, seq_len_k) -> (..., seq_len_q, seq_len_k)
    scores = torch.matmul(Q, K.transpose(-2, -1))

    # 2. 縮放 (Scaling)
    # 除以 sqrt(d_k) 穩定訓練過程
    scores = scores / math.sqrt(d_k)

    # 3. 應用遮罩 (Masking, GPT 關鍵)
    # 這就是 Self-Attention 在 GPT (解碼器) 中的重要差異：
    # 確保當前 Token i 只能注意到歷史 Token (j <= i)
    if mask is not None:
        # 將遮罩位置的分數設為一個極小的負數，經過 Softmax 後權重接近 0
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # 4. 轉換為注意力權重 (Softmax)
    # 這是動態計算出來的相關性分數 (用完即丟的 activations)
    attn_weights = F.softmax(scores, dim=-1)

    # 5. 加權求和 (與 V 相乘)
    # (..., seq_len_q, seq_len_k) @ (..., seq_len_k, d_v) -> (..., seq_len_q, d_v)
    output = torch.matmul(attn_weights, V)

    return output, attn_weights

# --- 2. Multi-Head Attention (多頭注意力) 層 ---
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model          # 模型嵌入維度 (d_model)
        self.num_heads = num_heads      # 頭數 (h)
        self.head_dim = d_model // num_heads # 單個 Head 的維度 (d_k = d_model / h)

        # 1. Q, K, V 投影層 (Linear Layer)
        # 參數數量計算：3 * (d_model * d_model)
        # 這裡將 Q, K, V 合併在一個層中，然後拆分，效率更高
        self.W_qkv = nn.Linear(d_model, 3 * d_model, bias=False)

        # 2. 輸出投影層 (Linear Layer)
        # 參數數量計算：d_model * d_model
        # 這是將 h 個 Head 拼接後的結果重新映射回 d_model
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None, kv_cache=None):
        # x: (batch_size, seq_len, d_model)
        
        B, T, C = x.size() # B=Batch, T=Seq_len, C=d_model

        # --- a. 投影 Q/K/V ---
        # 輸出: (B, T, 3 * d_model)
        qkv = self.W_qkv(x)
        
        # 拆分 Q, K, V (從 d_model 拆分成 3 個 d_model)
        # 並重塑成適合多頭計算的形狀 (B, T, num_heads, head_dim)
        q, k, v = qkv.chunk(3, dim=-1) # 沿最後一維拆分
        
        # 轉置：(B, num_heads, T, head_dim)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # --- b. 處理 KV Cache (加速推理) ---
        if kv_cache is not None:
             # K, V 矩陣就是我們儲存下來的 activations
             k_prev, v_prev = kv_cache
             k = torch.cat([k_prev, k], dim=2) # 沿序列長度維度拼接
             v = torch.cat([v_prev, v], dim=2)

        # --- c. 計算 Scaled Dot-Product Attention ---
        # 這裡計算出 Attention 權重 (動態相關性分數) 和 Output
        attn_output, _ = scaled_dot_product_attention(q, k, v, mask=mask)

        # --- d. 拼接與輸出投影 ---
        # 拼接 (Concat): 恢復成 (B, T, d_model) 形狀
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        
        # 輸出投影 (W_o)
        output = self.W_o(attn_output)
        return output, (k, v) # 返回 K, V 矩陣用於下一次的 Cache

# --- 3. Feed-Forward Network (前饋網路) ---
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        # d_ff 通常是 4 * d_model
        
        # 第一層：擴展維度 (d_model -> d_ff)
        # 參數數量：d_model * d_ff
        self.linear_in = nn.Linear(d_model, d_ff)
        
        # 第二層：縮小維度 (d_ff -> d_model)
        # 參數數量：d_ff * d_model
        self.linear_out = nn.Linear(d_ff, d_model)
        
        # GPT 通常使用 GELU 激活函數
        self.activation = nn.GELU()

    def forward(self, x):
        # 參數計算：(d_model * d_ff) + (d_ff * d_model) = 2 * d_model * d_ff
        x = self.linear_in(x)
        x = self.activation(x)
        x = self.linear_out(x)
        return x

# --- 4. GPT Decoder Block (核心區塊) ---
class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout_rate):
        super().__init__()
        
        # 核心組件 1: Multi-Head Self-Attention
        self.attn = MultiHeadAttention(d_model, num_heads)
        
        # 核心組件 2: Feed-Forward Network
        self.ffn = FeedForward(d_model, d_ff)

        # 結構組件: Layer Normalization (用於穩定訓練)
        # 參數數量：LayerNorm 只包含兩個可學習參數 (gamma, beta)，共 2 * d_model
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # 結構組件: Dropout (用於防止過擬合)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x, mask=None, kv_cache=None):
        
        # 1. 第一子層：Attention + Add & Norm
        
        # LayerNorm (Pre-norm 結構，GPT 常用)
        x_norm = self.norm1(x)
        
        # Multi-Head Attention (Self-Attention)
        attn_output, new_kv_cache = self.attn(x_norm, mask=mask, kv_cache=kv_cache)
        
        # Residual Connection (Add) + Dropout
        x = x + self.dropout1(attn_output)
        
        # 2. 第二子層：FFN + Add & Norm
        
        # LayerNorm
        x_norm = self.norm2(x)
        
        # Feed-Forward Network
        ffn_output = self.ffn(x_norm)
        
        # Residual Connection (Add) + Dropout
        x = x + self.dropout2(ffn_output)
        
        # 返回 Block 輸出和更新後的 KV Cache
        return x, new_kv_cache


# --- 範例運行與參數計算驗證 ---

d_model = 256     # 嵌入維度 H
num_heads = 8     # 頭數 h
d_ff = 4 * d_model # FFN 擴展維度 4H

print(f"--- 模型配置參數 ---")
print(f"d_model (H): {d_model}")
print(f"num_heads (h): {num_heads}")
print(f"d_ff (4H): {d_ff}\n")

# 建立一個 Block 實例
block = DecoderBlock(d_model, num_heads, d_ff, 0.1)

# 驗證參數計算
total_params = sum(p.numel() for p in block.parameters() if p.requires_grad)

# 1. Attention 層參數
# 4 * d_model^2 = 4 * 256^2 = 262144 (這裡因 W_qkv/W_o 皆含 bias 故略有增加)
attn_params = sum(p.numel() for name, p in block.named_parameters() if 'attn' in name)

# 2. FFN 層參數
# 2 * d_model * d_ff = 2 * 256 * 1024 = 524288 (這裡因 linear_in/out 皆含 bias 故略有增加)
ffn_params = sum(p.numel() for name, p in block.named_parameters() if 'ffn' in name)

# 3. LayerNorm 參數
# 2 * 2 * d_model = 4 * 256 = 1024
norm_params = sum(p.numel() for name, p in block.named_parameters() if 'norm' in name)


print(f"--- 單一 Block 參數分佈 ---")
print(f"Attention 層總參數 (QKV + Out): {attn_params}")
print(f"FFN 層總參數 (W_in + W_out): {ffn_params}")
print(f"LayerNorm 層總參數 (gamma/beta): {norm_params}")
print(f"**Block 總參數 (約 12 * d_model^2): {total_params}**\n")

# 模擬輸入
seq_len = 10
batch_size = 4
x = torch.randn(batch_size, seq_len, d_model)

# 模擬 Attention Mask (用於確保 Token i 只看 i 之前的所有 Token)
# 對角線及以下為 True (1)，以上為 False (0)
attn_mask = torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len)

# 執行前向傳播
output, kv_cache = block(x, mask=attn_mask)
print(f"--- 執行結果 ---")
print(f"輸入形狀: {x.shape}")
print(f"輸出形狀: {output.shape}")
print(f"KV Cache K 形狀 (用於下一輪推理加速): {kv_cache[0].shape}")