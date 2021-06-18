import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch.nn.modules.linear import Linear


#Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int = 768, num_patches: int = None, dropout: float = 0.):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, n):
        x = x + self.pos_embed[:, :(n+1)]
        x = self.dropout(x)
        return x

#Norm Layer
class Norm(nn.Module):
    def __init__(self, d_model: int = 768, next_layer: nn.Module = None):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.next_layer = next_layer
    def forward(self, x: torch.Tensor, **kwargs):
        x = self.norm(x)
        return self.next_layer(x, **kwargs)

#Feed Forward MLP
class FeedForward(nn.Module):
    def __init__(self, d_model: int = 768, d_mlp: int = 3072, dropout: float = 0.):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(d_model, d_mlp),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_mlp, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor):
        return self.layers(x)

#Multi-Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int = 768, n_head: int = 12, dropout: float = 0.):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.dropout = nn.Dropout(dropout)

        d_head = d_model // n_head
        project_out = not (n_head == 1 and d_head == d_model)

        self.scale = d_head ** -0.5
        self.softmax = nn.Softmax(dim = -1)
        self.w_qkv = nn.Linear(d_model, d_model * 3, bias = False)

        self.fc_out = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.n_head
        qkv = self.w_qkv(x).chunk(3, dim = -1)
        queries, keys, values = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        # Compute Attention score
        scores = torch.einsum('b h i d, b h j d -> b h i j', queries, keys) * self.scale
        attention = self.softmax(scores)

        x = torch.einsum('b h i j, b h j d -> b h i d', attention, values)
        x = rearrange(x, 'b h n d -> b n (h d)')

        return x





