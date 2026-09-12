from __future__ import annotations

import torch
from torch import nn


class SafeEmbedding(nn.Module):
    """沿用 PPO 的 ID 嵌入语义，取消对旧 DQN 项目的导入。"""

    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size + 2, embedding_dim, padding_idx=vocab_size + 1)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        ids = ids.long()
        mapped = torch.where((ids < 0) | (ids >= self.vocab_size), self.vocab_size, ids)
        return self.embedding(torch.where(ids == -1, self.vocab_size + 1, mapped))

