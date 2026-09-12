from __future__ import annotations

import torch
from torch import nn


def resource_output_dim(cfg):
    return 2 * 4 * (cfg["skill_embedding_dim"] + 1)


class SkillSlotEmbedding(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.variant = nn.Embedding(4, cfg["skill_embedding_dim"], padding_idx=0)

    def forward(self, value, mask):
        value = torch.where(mask, value, torch.zeros_like(value))
        embedded = self.variant(value) * mask.unsqueeze(-1)
        return torch.cat((embedded, mask.unsqueeze(-1).to(embedded.dtype)), -1)


class ResourceEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.output_dim = resource_output_dim(cfg)
        # 每槽区分必杀类型，双方共享同槽嵌入；等级和卡牌没有任何前向支路。
        self.skill_slots = nn.ModuleDict({
            f"skill_slot_{i}": SkillSlotEmbedding(cfg) for i in range(1, 5)
        })

    def forward(self, observation):
        features = []
        for side in ("self", "opponent"):
            values = observation[f"{side}_skill_categorical"]
            masks = observation[f"{side}_skill_mask"]
            for index, module in enumerate(self.skill_slots.values()):
                features.append(module(values[..., index], masks[..., index]))
        return torch.cat(features, -1)

