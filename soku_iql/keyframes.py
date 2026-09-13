from __future__ import annotations

import torch

from .bc_core.action_space import ACTION_COUNT


@torch.no_grad()
def build_changepoint_mask(expert_actions, valid_mask, previous_expert_actions):
    """复用 Dataset 在完整轨迹上对齐的真实历史；不在 batch 内自行右移或猜测连续性。"""
    if expert_actions.shape != valid_mask.shape or previous_expert_actions.shape != expert_actions.shape:
        raise ValueError("关键帧标签、有效 mask 和上一帧专家动作必须完全同形状")
    if (expert_actions.dtype != torch.long or previous_expert_actions.dtype != torch.long
            or valid_mask.dtype != torch.bool):
        raise ValueError("关键帧动作必须为 int64，有效 mask 必须为 bool")
    if expert_actions.device != valid_mask.device or previous_expert_actions.device != expert_actions.device:
        raise ValueError("关键帧标签、有效 mask 和真实历史必须位于同一设备")
    # episode/终局/断帧边界已由 previous_actions 标成 START=144。
    # 切片首帧若带有合法历史则照常比较；无历史的有效帧仍参与普通 CE。
    changepoint_valid_mask = valid_mask & (previous_expert_actions >= 0) & (previous_expert_actions < ACTION_COUNT)
    is_changepoint = changepoint_valid_mask & (expert_actions != previous_expert_actions)
    return is_changepoint, changepoint_valid_mask
