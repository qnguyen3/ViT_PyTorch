from module.Layers import *
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch.nn.modules.linear import Linear

#Vision Transformer
class ViT(nn.Module):
    def __init__(self, img_size: int = 256, patch_size: int = 16, 
                num_class: int = 1000, d_model: int = 768, n_head: int = 12, 
                n_layers:int = 12, d_mlp: int = 3072, channels: int = 3, 
                dropout: float = 0., pool: str = 'cls'):
        super().__init__()

        img_h, img_w = img_size, img_size
        patch_h, patch_w = patch_size, patch_size

        assert img_h % patch_h == 0, 'image dimension must be divisible by patch dimension'
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        num_patches = (img_h // patch_h) * (img_w // patch_w)
        patch_dim = channels * patch_h * patch_w

        self.patches_embed = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_h, p2 = patch_w),
            nn.Linear(patch_dim, d_model)
        )

        self.pos_embed = PositionalEncoding(d_model, num_patches, dropout)
        self.class_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pool = pool
    
        self.transformer = Transformer(d_model, n_head, n_layers, d_mlp, dropout)
        self.dropout = nn.Dropout(dropout)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_class)
        )
    
    def forward(self, img):
        x = self.patches_embed(img)
        b, n, _ = x.shape
        class_token = repeat(self.class_token, '() n d -> b n d', b = b)
        #Concat Class Token with image patches
        x = torch.cat((class_token,x), dim=1)
        #Add Positional Encoding
        x = self.pos_embed(x, n)
        x = self.transformer(x)
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        #MLP Head
        x = self.mlp_head(x)
        return x

# Transformer
class Transformer(nn.Module):
    def __init__(self, d_model: int = 768, n_head: int = 12, n_layers:int = 12,
                d_mlp: int = 3072, dropout: float = 0.):
        super().__init__()

        self.block = nn.ModuleList([
            Norm(d_model, MultiHeadAttention(d_model, n_head, dropout)),
            Norm(d_model, FeedForward(d_model, d_mlp, dropout))
            ])
        self.layers = nn.ModuleList([self.block for _ in range(n_layers)])

    def forward(self, x):
        for attention, mlp in self.layers:
            x = attention(x) + x
            x = mlp(x) + x
        return x
