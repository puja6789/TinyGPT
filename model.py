import torch
import torch.nn as nn
import torch.nn.functional as F
import config
from transformer_blocks import Block


class TinyGPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, config.embedding_dim)
        self.position_embedding = nn.Embedding(config.block_size, config.embedding_dim)

        self.blocks = nn.Sequential(
            *[Block(config.embedding_dim, config.block_size, config.n_heads)
              for _ in range(config.n_layers)]
        )

        self.ln_f = nn.LayerNorm(config.embedding_dim)
        self.head = nn.Linear(config.embedding_dim, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok = self.token_embedding(idx)
        pos = self.position_embedding(torch.arange(T, device=idx.device))

        x = tok + pos
        x = self.blocks(x)
        x = self.ln_f(x)

        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(B * T, -1),
                targets.view(B * T)
            )

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -config.block_size:]

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, 1)

            idx = torch.cat((idx, next_idx), dim=1)

        return idx