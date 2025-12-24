import torch
import torch.nn as nn

class ViT(nn.Module):
    def __init__(self, image_size=28, patch_size=7, num_classes=10, dim=64, depth=6, heads=8, mlp_dim=128, channels=1):
        super().__init__()
        assert image_size % patch_size == 0, "Image dimensions must be divisible by the patch size."
        
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2

        # 1. Patch Embedding: 將圖片切塊並投影到指定維度
        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(channels, dim, kernel_size=patch_size, stride=patch_size),
            nn.Flatten(2),
        )

        # 2. Learnable Tokens: Class token 與 Position embedding
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        
        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=mlp_dim, 
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # 4. MLP Head: 最後的分類層
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        # x shape: (batch, channels, h, w) -> (batch, dim, n_patches)
        x = self.to_patch_embedding(img) 
        x = x.transpose(1, 2) # (batch, n_patches, dim)

        # 加入 Class Token
        b, n, _ = x.shape
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # 加入位置資訊
        x += self.pos_embedding

        # 進入 Transformer
        x = self.transformer_encoder(x)

        # 取出 Class Token 的輸出做分類
        return self.mlp_head(x[:, 0])

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
