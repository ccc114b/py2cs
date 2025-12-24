import torch
import torch.nn as nn
import math

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t):
        half_dim = self.mlp[0].in_features // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return self.mlp(emb)

class DiTBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.time_mlp = nn.Linear(dim, dim)

    def forward(self, x, t_emb):
        x = x + self.time_mlp(t_emb).unsqueeze(1)
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

class TinyDiT(nn.Module):
    def __init__(self, image_size=28, patch_size=7, dim=64, depth=3, heads=4):
        super().__init__()
        self.patch_size = patch_size
        num_patches = (image_size // patch_size) ** 2
        
        self.to_patch = nn.Conv2d(1, dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.time_embedding = TimeEmbedding(dim)
        
        self.blocks = nn.ModuleList([DiTBlock(dim, heads) for _ in range(depth)])
        self.linear_final = nn.Linear(dim, patch_size ** 2)

    def forward(self, x, t):
        b, c, h, w = x.shape
        x = self.to_patch(x).flatten(2).transpose(1, 2)
        x += self.pos_embedding
        t_emb = self.time_embedding(t)
        
        for block in self.blocks:
            x = block(x, t_emb)
            
        x = self.linear_final(x)
        p = self.patch_size
        x = x.view(b, h//p, w//p, p, p).permute(0, 1, 3, 2, 4).contiguous()
        return x.view(b, 1, 28, 28)
    
def get_device():
    if torch.cuda.is_available():
        print("Using CUDA")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("Using MPS")
        return torch.device("mps")
    else:
        print("Using CPU")
        return torch.device("cpu")
