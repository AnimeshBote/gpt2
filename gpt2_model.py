import os
import torch
from torch.utils.data import IterableDataset, DataLoader
from tokenizers import ByteLevelBPETokenizer
import torch.nn as nn
from flash import FlashBlock

# 3. Manual GPT-2-like model
# class GPT2LikeModel(nn.Module):
#     def __init__(self, vocab_size, block_size, n_embd=768, n_layer=6, n_head=6, dropout=0.1):
#         super().__init__()
#         self.token_emb = nn.Embedding(vocab_size, n_embd)
#         self.pos_emb = nn.Embedding(block_size, n_embd)
#         encoder_layer = nn.TransformerEncoderLayer(d_model=n_embd, nhead=n_head, batch_first=False)
#         self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
#         self.ln_f = nn.LayerNorm(n_embd)
#         self.head = nn.Linear(n_embd, vocab_size)
#         # self.register_buffer('causal_mask', nn.Transformer.generate_square_subsequent_mask(block_size))
#         self.drop = nn.Dropout(0.1)

#     def forward(self, idx):
#         b, t = idx.size()
#         # print(t)
#         device = idx.device
#         tok_emb = self.token_emb(idx)  # [b, t, n_embd]
#         pos = torch.arange(t, device=device).unsqueeze(0)  # [1, t]
#         pos_emb = self.pos_emb(pos)  # [1, t, n_embd]
#         x = self.drop(tok_emb + pos_emb)  # [b, t, n_embd]
#         x = x.transpose(0, 1)  # [t, b, n_embd]

#         # ✅ Create causal mask
#         mask = nn.Transformer.generate_square_subsequent_mask(t).to(device)  # [t, t]
#         # mask = self.causal_mask[:t, :t].to(device)

#         x = self.blocks(x, mask=mask)  # [t, b, n_embd]
#         x = x.transpose(0, 1)          # [b, t, n_embd]
#         x = self.ln_f(x)
#         logits = self.head(x)          # [b, t, vocab_size]
#         return logits

class GPT2LikeModel(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd=768, n_layer=6, n_head=6, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([
            FlashBlock(n_embd, n_head, dropout) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)
        self.drop = nn.Dropout(dropout)

    def forward(self, idx):
        b, t = idx.size()
        device = idx.device
        
        # FIXED: Clone embeddings to prevent CUDA graph memory overwrites
        tok_emb = self.token_emb(idx)
        pos = torch.arange(t, device=device).unsqueeze(0)
        pos_emb = self.pos_emb(pos)
        
        x = self.drop(tok_emb + pos_emb)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        return self.head(x)