from __future__ import annotations

import copy

import torch
from torch import nn

from .action_space import ACTION_COUNT, ACTION_SCHEMA
from .config import MODEL_DEFAULTS, active_modules, network_version_for
from .nn_modules import (
    CurrentStateEncoder,
    FusionEncoder,
    ObjectSetEncoder,
    current_state_input_dim,
    initialize,
)
from .schema import policy_input_manifest
from .temporal import TemporalConvEncoder


def network_spec(cfg):
    unknown = set(cfg) - set(MODEL_DEFAULTS)
    if unknown:
        raise ValueError(f"模型包含旧架构或未知字段：{sorted(unknown)}")
    cfg = {**MODEL_DEFAULTS, **cfg}
    input_dim = current_state_input_dim(cfg)
    if input_dim != 228:
        raise ValueError(f"当前网络版本固定 228D 状态输入，实际 schema 生成 {input_dim}D")
    return {
        "network_version": network_version_for(cfg),
        "model": copy.deepcopy(cfg),
        "inputs": policy_input_manifest(),
        "action_schema": ACTION_SCHEMA,
        "output_semantics": "categorical_logits",
        "current": {"input_dim": input_dim, "output_dim": cfg["current_hidden_dim"], "layers": 1},
        "temporal": {
            "mode": "tcn",
            "input_dim": input_dim,
            "input_semantics": "last_32_expanded_state_features",
            "output_dim": cfg["temporal_output_dim"],
            "context_frames": TemporalConvEncoder.context_frames,
            "convolutions_per_block": 2,
            "kernel_size": 2,
            "causal_stem": True,
            "dilations": [1, 2, 4, 8],
            "receptive_field_frames": 32,
            "gru_removed": True,
        },
        "fusion": {
            "input_dim": cfg["current_hidden_dim"] + cfg["temporal_output_dim"]
            + 2 * cfg["object_set_dim"],
            "hidden_dim": cfg["fusion_dim"],
            "output_dim": ACTION_COUNT,
        },
    }


class BCNetwork(nn.Module):
    """精简状态分支加独立 TCN32 的 Joint144 行为克隆网络。"""

    def __init__(self, cfg: dict, network_version: str | None = None):
        super().__init__()
        unknown = set(cfg) - set(MODEL_DEFAULTS)
        if unknown:
            raise ValueError(f"模型包含旧架构或未知字段：{sorted(unknown)}")
        cfg = {**MODEL_DEFAULTS, **cfg}
        expected_version = network_version_for(cfg)
        if network_version is not None and network_version != expected_version:
            raise ValueError(
                f"BC 网络结构不兼容：{network_version}，当前配置需要 {expected_version}；"
                "旧 Joint432/GRU checkpoint 不能部分加载"
            )
        for key in (
            "current_hidden_dim",
            "object_hidden_dim",
            "object_set_dim",
            "temporal_hidden_dim",
            "temporal_output_dim",
            "fusion_dim",
        ):
            if cfg[key] != MODEL_DEFAULTS[key]:
                raise ValueError(f"当前 TCN32 架构固定 {key}={MODEL_DEFAULTS[key]}")

        self.uses_resources = True
        self.temporal_mode = "tcn"
        self.current_encoder = CurrentStateEncoder(cfg)
        if self.current_encoder.input_dim != 228:
            raise ValueError(
                f"当前网络版本要求 228D 状态输入，实际为 {self.current_encoder.input_dim}D；"
                "状态 schema 变化后必须升级 network_version"
            )
        self.object_encoder = ObjectSetEncoder(cfg)
        self.tcn = TemporalConvEncoder(
            self.current_encoder.input_dim,
            cfg["temporal_hidden_dim"],
            cfg["temporal_output_dim"],
        )
        self.fusion = FusionEncoder(cfg)
        self.policy_head = nn.Linear(cfg["fusion_dim"], ACTION_COUNT)
        self.memory_dim = self.current_encoder.input_dim
        self.spec = network_spec(cfg)

        self.apply(initialize)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.zeros_(self.policy_head.bias)

    def module_groups(self):
        names = ("current_encoder", "object_encoder", "tcn", "fusion", "policy_head")
        return {name: getattr(self, name) for name in names}

    def module_status(self):
        active = set(active_modules(self.spec["model"]))
        return {
            name: {
                "active": name in active,
                "frozen": not any(parameter.requires_grad for parameter in module.parameters()),
                "bypassed": name not in active,
            }
            for name, module in self.module_groups().items()
        }

    def state_features(self, obs):
        """构造 228D 状态向量；此结果同时送入当前帧分支和独立历史分支。"""
        return self.current_encoder.features(obs)

    def encode_objects(self, obs):
        return [
            self.object_encoder(
                obs[f"{side}_object_numerical"],
                obs[f"{side}_object_categorical"],
                obs[f"{side}_object_mask"],
                side,
            )
            for side in ("self", "opponent")
        ]

    @staticmethod
    def _main_observation(obs, burn_in):
        return {key: value[:, burn_in:] for key, value in obs.items()}

    @staticmethod
    def _align_prefix(features, burn_in, burn_lengths):
        mask = torch.ones(features.shape[:2], dtype=torch.bool, device=features.device)
        if not burn_in:
            return features, mask
        if burn_in != 31 or burn_lengths is None:
            raise ValueError("TCN32 训练需要 31 帧前导上下文和每条样本的真实历史长度")
        lengths = burn_lengths.to(features.device).long()
        offsets = torch.arange(burn_in, device=features.device)[None] - (burn_in - lengths[:, None])
        prefix_mask = offsets >= 0
        gather = offsets.clamp_min(0)[..., None].expand(-1, -1, features.shape[-1])
        # Dataset 前导区是右侧 padding；重新对齐成左 padding，使最近历史紧邻监督帧。
        prefix = features[:, :burn_in].gather(1, gather) * prefix_mask[..., None]
        mask[:, :burn_in] = prefix_mask
        return torch.cat((prefix, features[:, burn_in:]), dim=1), mask

    def _fuse(self, current, temporal, objects):
        feature = torch.cat((current, temporal, *objects), dim=-1)
        return self.policy_head(self.fusion(feature))

    def forward(self, obs, burn_in: int = 0, burn_lengths=None, *, return_aux=False):
        state = self.state_features(obs)
        state, history_mask = self._align_prefix(state, burn_in, burn_lengths)
        main_state = state[:, burn_in:]
        current = self.current_encoder.forward_features(main_state)
        temporal = self.tcn(state, history_mask)[:, burn_in:]
        objects = self.encode_objects(self._main_observation(obs, burn_in))
        logits = self._fuse(current, temporal, objects)
        # PALR 只读取监督段的 TCN 输出；默认推理路径和所有网络参数保持原样。
        return (logits, {"temporal_feature": temporal}) if return_aux else logits

    @torch.no_grad()
    def act(self, obs, memory=None):
        logits, memory = self.step_logits(obs, memory)
        return logits.argmax(-1), memory

    @torch.no_grad()
    def step_logits(self, obs, memory=None):
        """单帧接口保留此前 31 个 228D 状态，与当前帧组成 32 帧窗口。"""
        state = self.state_features(obs)
        if state.ndim != 2:
            raise ValueError("单帧推理 observation 必须生成 [batch,228] 状态")
        if memory is not None and (
            memory.ndim != 3
            or memory.shape[0] != state.shape[0]
            or memory.shape[1] > 31
            or memory.shape[2] != self.memory_dim
        ):
            raise ValueError(f"TCN32 历史必须为 [batch,最多31帧,{self.memory_dim}]")
        history = state[:, None] if memory is None else torch.cat((memory, state[:, None]), dim=1)
        temporal = self.tcn(history)[:, -1]
        current = self.current_encoder.forward_features(state)
        return self._fuse(current, temporal, self.encode_objects(obs)), history[:, -31:].detach()

