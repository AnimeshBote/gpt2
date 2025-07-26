import torch
from flash_attn.modules.mha import MHA
# from flash_attn.modules.mlp import MLP
from torch.nn import LayerNorm
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, dim, hidden_dim=None, dropout=0.1):
        super().__init__()
        hidden_dim = hidden_dim or 4 * dim
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.fc2(self.act(self.fc1(x))))

class FlashBlock(nn.Module):
    def __init__(self, dim, n_heads, dropout=0.1):
        super().__init__()
        self.ln1 = LayerNorm(dim)
        self.attn = MHA(
            embed_dim=dim,
            num_heads=n_heads,
            dropout=dropout,
            causal=True,  # very important
        )
        self.ln2 = LayerNorm(dim)
        self.mlp = MLP(
            dim=dim,
            dropout=dropout,
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
