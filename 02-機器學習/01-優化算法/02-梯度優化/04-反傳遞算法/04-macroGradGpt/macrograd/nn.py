import numpy as np
from macrograd.engine import Tensor

class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.grad)
    def parameters(self):
        return []

class Embedding(Module):
    def __init__(self, vocab_size, embed_dim):
        self.weight = Tensor(np.random.randn(vocab_size, embed_dim) * 0.01)
    def __call__(self, idx):
        # idx 是 (batch, seq_len) 的整數陣列
        # 透過 one-hot 矩陣相乘來模擬 lookup (以便於自動微分)
        one_hot = np.eye(self.weight.shape[0])[idx] # (B, T, V)
        return Tensor(one_hot).matmul(self.weight)
    def parameters(self):
        return [self.weight]

class LayerNorm(Module):
    def __init__(self, dim):
        self.gamma = Tensor(np.ones((1, 1, dim)))
        self.beta = Tensor(np.zeros((1, 1, dim)))
    def __call__(self, x):
        mean = x.sum(axis=-1, keepdims=True) * (1.0/x.shape[-1])
        var = ((x - mean)**2).sum(axis=-1, keepdims=True) * (1.0/x.shape[-1])
        x_hat = (x - mean) * ((var + 1e-5)**-0.5)
        return self.gamma * x_hat + self.beta
    def parameters(self):
        return [self.gamma, self.beta]

class CausalSelfAttention(Module):
    def __init__(self, n_embd, n_head):
        self.n_head = n_head
        self.key = Linear(n_embd, n_embd)
        self.query = Linear(n_embd, n_embd)
        self.value = Linear(n_embd, n_embd)
        self.proj = Linear(n_embd, n_embd)

    def __call__(self, x):
        B, T, C = x.shape
        k = self.key(x).reshape(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).reshape(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).reshape(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Attention
        att = q.matmul(k.transpose(-2, -1)) * ((C // self.n_head)**-0.5)
        
        # Causal mask (numpy 實作)
        mask = np.tril(np.ones((T, T))) == 0
        att.data[:, :, mask] = -1e9 # 遮蓋未來資訊
        
        # Softmax 手動應用在最後一個維度
        # 由於 engine 的 softmax 較簡單，我們簡化處理
        att_exp = (att - Tensor(np.max(att.data, axis=-1, keepdims=True))).data
        att_exp = np.exp(att_exp)
        att_prob = Tensor(att_exp / np.sum(att_exp, axis=-1, keepdims=True))

        y = att_prob.matmul(v)
        y = y.transpose(1, 2).reshape(B, T, C)
        return self.proj(y)

    def parameters(self):
        return self.key.parameters() + self.query.parameters() + \
               self.value.parameters() + self.proj.parameters()

class Linear(Module):
    def __init__(self, nin, nout):
        self.W = Tensor(np.random.randn(nin, nout) * 0.02)
        self.b = Tensor(np.zeros(nout))
    def __call__(self, x):
        return x.matmul(self.W) + self.b
    def parameters(self):
        return [self.W, self.b]

class Block(Module):
    def __init__(self, n_embd, n_head):
        self.ln1 = LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.ln2 = LayerNorm(n_embd)
        self.mlp = MLP(n_embd, [4 * n_embd, n_embd])
    def __call__(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
    def parameters(self):
        return self.ln1.parameters() + self.attn.parameters() + \
               self.ln2.parameters() + self.mlp.parameters()

class MLP(Module):
    def __init__(self, nin, nouts):
        self.l1 = Linear(nin, nouts[0])
        self.l2 = Linear(nouts[0], nouts[1])
    def __call__(self, x):
        return self.l2(self.l1(x).tanh())
    def parameters(self):
        return self.l1.parameters() + self.l2.parameters()

class GPT(Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size):
        self.tok_emb = Embedding(vocab_size, n_embd)
        self.pos_emb = Tensor(np.random.randn(1, block_size, n_embd) * 0.01)
        self.blocks = [Block(n_embd, n_head) for _ in range(n_layer)]
        self.ln_f = LayerNorm(n_embd)
        self.head = Linear(n_embd, vocab_size)
        self.block_size = block_size

    def __call__(self, idx):
        B, T = idx.shape
        x = self.tok_emb(idx) + Tensor(self.pos_emb.data[:, :T, :])
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    def parameters(self):
        params = self.tok_emb.parameters() + [self.pos_emb]
        for block in self.blocks: params += block.parameters()
        params += self.ln_f.parameters() + self.head.parameters()
        return params
