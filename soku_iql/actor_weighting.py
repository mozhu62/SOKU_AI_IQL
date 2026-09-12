"""Actor 类别加权：保留全部有效帧，优势权重不参与分母。"""
import torch

from .bc_core.action_space import NEUTRAL_ACTION_ID


def weighted_actor_loss(ce, advantages, actions, neutral_weight=1.0):
    if ce.shape != advantages.shape or ce.shape != actions.shape:
        raise ValueError("CE、优势权重和动作必须同形状")
    if not 0 < neutral_weight <= 1:
        raise ValueError("Neutral 权重必须在 (0,1] 内")
    category = torch.where(actions == NEUTRAL_ACTION_ID, neutral_weight, 1.0)
    # 只按类别权重归一化；保留 IQL 优势权重的绝对尺度。
    return (category * advantages.detach() * ce).sum() / category.sum()
