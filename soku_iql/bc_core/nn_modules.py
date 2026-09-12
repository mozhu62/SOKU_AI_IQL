from __future__ import annotations

import math

import torch
from torch import nn

from .embeddings import SafeEmbedding
from .action_space import PREVIOUS_ACTION_VOCAB, START_ACTION_ID
from .resource_encoder import ResourceEncoder, resource_output_dim

from .schema import STATE_CONTINUOUS_FEATURES, OBJECT_NUMERICAL_FEATURES
from .schema import TACTICAL_FEATURES


def hidden_mlp(input_dim: int, hidden_dim: int, layers: int = 2) -> nn.Sequential:
    modules = []
    for _ in range(layers):
        modules.extend((nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU()))
        input_dim = hidden_dim
    return nn.Sequential(*modules)


def current_state_input_dim(cfg: dict) -> int:
    return (len(STATE_CONTINUOUS_FEATURES) + len(TACTICAL_FEATURES)
            + 2 * cfg["action_embedding_dim"] + 2 * cfg["block_embedding_dim"]
            + cfg["weather_embedding_dim"] + resource_output_dim(cfg)
            + cfg["previous_action_embedding_dim"] + 1 + 4 + 4)


def initialize(module: nn.Module) -> None:
    """编码层统一随机初始化；策略 logits 末层在模型中单独设置 gain。"""
    if isinstance(module, (nn.Linear, nn.Conv1d)):
        nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=0.02)
        if module.padding_idx is not None:
            with torch.no_grad():
                module.weight[module.padding_idx].zero_()


class CurrentStateEncoder(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.action = SafeEmbedding(cfg["action_vocab_size"], cfg["action_embedding_dim"])
        self.block = SafeEmbedding(cfg["block_vocab_size"], cfg["block_embedding_dim"])
        self.weather = SafeEmbedding(cfg["weather_vocab_size"], cfg["weather_embedding_dim"])
        self.previous_action = nn.Embedding(PREVIOUS_ACTION_VOCAB, cfg["previous_action_embedding_dim"],
                                            padding_idx=START_ACTION_ID)
        self.resources = ResourceEncoder(cfg)
        self.input_dim = current_state_input_dim(cfg)
        # 228D 状态映射为 256D；历史 TCN 仍独立接收逐帧状态向量。
        self.network = hidden_mlp(self.input_dim, cfg["current_hidden_dim"], layers=1)

    def features(self, obs) -> torch.Tensor:
        numeric, tactical, categorical = obs["state_continuous"], obs["tactical_state"], obs["state_categorical"]
        # 类别列序沿用 BC 分片清单：己方动作/动作段、敌方动作/动作段、生效天气。
        embedded = torch.cat((self.action(categorical[..., 0]), self.block(categorical[..., 1]),
                              self.action(categorical[..., 2]), self.block(categorical[..., 3]),
                              self.weather(categorical[..., 4])), dim=-1)
        features = torch.cat((numeric, tactical, embedded, self.resources(obs),
                              self.previous_action(obs["previous_joint_action_id"]),
                              obs["previous_action_duration"], obs["state_optional_continuous"],
                              obs["state_optional_mask"].to(numeric.dtype)), dim=-1)
        if features.shape[-1] != self.input_dim:
            raise ValueError(f"当前状态拼接维度应为 {self.input_dim}，实际为 {features.shape[-1]}")
        return features

    def forward_features(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != self.input_dim:
            raise ValueError(f"当前状态编码器需要 {self.input_dim}D 输入")
        return self.network(features)

    def forward(self, obs) -> torch.Tensor:
        return self.forward_features(self.features(obs))


class ObjectSetEncoder(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.mode = cfg["object_embedding_mode"]
        sides = ("self", "opponent") if self.mode == "separate" else ("shared",)
        self.action_embeddings = nn.ModuleDict({
            side: SafeEmbedding(cfg["action_vocab_size"], cfg["action_embedding_dim"]) for side in sides
        })
        self.block_embeddings = nn.ModuleDict({
            side: SafeEmbedding(cfg["block_vocab_size"], cfg["block_embedding_dim"]) for side in sides
        })
        self.output_dim = cfg["object_set_dim"]
        self.network = hidden_mlp(len(OBJECT_NUMERICAL_FEATURES) + cfg["action_embedding_dim"] + cfg["block_embedding_dim"], cfg["object_hidden_dim"])

    def forward(self, numeric: torch.Tensor, categorical: torch.Tensor, mask: torch.Tensor, side: str) -> torch.Tensor:
        if (numeric.ndim < 3 or numeric.shape[-1] != len(OBJECT_NUMERICAL_FEATURES) or
                categorical.shape != (*numeric.shape[:-1], 2)):
            raise ValueError("对象输入应为 [..., N, 8] 数值和 [..., N, 2] 类别")
        if side not in ("self", "opponent") or mask.dtype != torch.bool or mask.shape != numeric.shape[:-1]:
            raise ValueError("对象侧别或对象 mask 形状不正确")
        if numeric.shape[-2] == 0:
            return numeric.new_zeros((*numeric.shape[:-2], self.output_dim))
        # 在进入 MLP 前清除 padding；槽位垃圾值即使为 NaN/Inf，也不能污染输出或梯度。
        numeric = torch.where(mask[..., None], numeric, torch.zeros_like(numeric))
        categorical = torch.where(mask[..., None], categorical, torch.zeros_like(categorical))
        key = side if self.mode == "separate" else "shared"
        embedded = torch.cat((self.action_embeddings[key](categorical[..., 0]),
                              self.block_embeddings[key](categorical[..., 1])), dim=-1)
        entity = self.network(torch.cat((numeric, embedded), dim=-1))
        entity = torch.where(mask[..., None], entity, torch.zeros_like(entity))
        mean = entity.sum(dim=-2) / mask.sum(dim=-1, keepdim=True).clamp_min(1).to(entity.dtype)
        maximum = entity.masked_fill(~mask[..., None], torch.finfo(entity.dtype).min).amax(dim=-2)
        maximum = torch.where(mask.any(dim=-1, keepdim=True), maximum, torch.zeros_like(maximum))
        return torch.cat((mean, maximum), dim=-1)


class FusionEncoder(nn.Sequential):
    def __init__(self, cfg: dict) -> None:
        input_dim = cfg["current_hidden_dim"] + cfg["temporal_output_dim"] + 2 * cfg["object_set_dim"]
        super().__init__(*hidden_mlp(input_dim, cfg["fusion_dim"], layers=1))

