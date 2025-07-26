from flash_attn.modules.mha import MHA
from flash_attn.modules.mlp import MLP
from torch.nn import LayerNorm

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
            hidden_size=dim,
            dropout=dropout,
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
