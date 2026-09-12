"""只筛选 Actor 监督位置，不改轨迹、时序输入或 Q/V 样本。"""
from fractions import Fraction

import torch

from .bc_core.action_space import NEUTRAL_ACTION_ID


def build_actor_mask(actions, valid_mask, enabled=False, neutral_max_fraction=0.25):
    if actions.shape != valid_mask.shape:
        raise ValueError("Actor 动作与有效 mask 必须同形状")
    if not 0 <= neutral_max_fraction <= 1:
        raise ValueError("Neutral 上限必须位于 [0, 1]")
    selected = valid_mask.bool().clone()
    neutral = selected & (actions == NEUTRAL_ACTION_ID)
    total, count = int(selected.sum()), int(neutral.sum())
    keep = count
    if enabled and neutral_max_fraction < 1:
        # k/(非 Neutral 数+k) <= p；用十进制有理数避免向下取整边界误差。
        ratio = Fraction(str(neutral_max_fraction))
        keep = min(count, (total-count)*ratio.numerator // (ratio.denominator-ratio.numerator))
        if keep < count:
            selected[neutral] = False
            indices = neutral.reshape(-1).nonzero(as_tuple=False).squeeze(-1)
            chosen = indices[torch.randperm(count, device=actions.device)[:keep]]
            selected.view(-1)[chosen] = True
    remaining = total-count+keep
    return selected, dict(actor_candidate_samples=total, actor_samples=remaining,
                          actor_neutral_before=count, actor_neutral_after=keep,
                          actor_filtered_samples=count-keep,
                          actor_neutral_fraction_before=count/total if total else None,
                          actor_neutral_fraction_after=keep/remaining if remaining else None,
                          actor_sampling_enabled=enabled, actor_neutral_max_fraction=neutral_max_fraction)
