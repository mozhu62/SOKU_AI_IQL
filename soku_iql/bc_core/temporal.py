from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class CausalResidualBlock(nn.Module):
    """两层同膨胀率因果卷积；LayerNorm 只在单帧通道维上统计。"""

    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.dilation = dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=2, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=2, dilation=dilation)
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)

    def _causal(self, convolution, values):
        return convolution(F.pad(values.transpose(1, 2), (self.dilation, 0))).transpose(1, 2)

    def forward(self, values, mask):
        temporal = F.silu(self.norm1(self._causal(self.conv1, values)))
        temporal = temporal * mask[..., None].to(temporal.dtype)
        temporal = self.norm2(self._causal(self.conv2, temporal))
        result = F.silu(values + temporal)
        return result * mask[..., None].to(result.dtype)


class TemporalConvEncoder(nn.Module):
    """独立读取 32 帧原始状态的宽 TCN，不再接收 Current Encoder 的压缩结果。"""

    context_frames = 32
    output_dim = 256

    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 256):
        super().__init__()
        if output_dim != hidden_dim:
            raise ValueError("当前 TCN32 要求 hidden_dim 与 output_dim 一致")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.projection = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU())
        # stem 增加一个历史间隔；四个双卷积残差块再增加 30 个间隔，最终严格覆盖 32 帧。
        self.stem = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=2)
        self.stem_norm = nn.LayerNorm(hidden_dim)
        self.blocks = nn.ModuleList(CausalResidualBlock(hidden_dim, dilation) for dilation in (1, 2, 4, 8))

    def forward(self, features, mask=None):
        if mask is None:
            mask = torch.ones(features.shape[:2], dtype=torch.bool, device=features.device)
        if features.shape[-1] != self.input_dim:
            raise ValueError(f"TCN32 需要每帧 {self.input_dim}D 状态输入")
        if mask.shape != features.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("TCN 有效帧 mask 必须与 [batch,time] 一致")
        # 每层都抹去左侧 padding，防止偏置在假历史上生成可传播的伪特征。
        values = self.projection(features) * mask[..., None].to(features.dtype)
        stem = self.stem(F.pad(values.transpose(1, 2), (1, 0))).transpose(1, 2)
        values = F.silu(values + self.stem_norm(stem)) * mask[..., None].to(values.dtype)
        for block in self.blocks:
            values = block(values, mask)
        return values

