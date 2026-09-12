from __future__ import annotations

import copy

import torch
from torch import nn

from .bc_core.models import BCNetwork


class IQLNetworks(nn.Module):
    def __init__(self, model_config, actor_state=None):
        super().__init__()
        self.actor = BCNetwork(model_config)
        if actor_state is not None:
            self.actor.load_state_dict(actor_state, strict=True)
        # Q/V 的编码器从 BC 初始化，但参数独立；不能把 BC logits 当作已学好的 Q 值。
        self.q1, self.q2, self.value = [copy.deepcopy(self.actor) for _ in range(3)]
        for network, width in ((self.q1, 144), (self.q2, 144), (self.value, 1)):
            network.policy_head = nn.Linear(model_config["fusion_dim"], width)
            nn.init.orthogonal_(network.policy_head.weight, gain=.01)
            nn.init.zeros_(network.policy_head.bias)
        self.target_q1, self.target_q2 = copy.deepcopy(self.q1), copy.deepcopy(self.q2)
        for target in (self.target_q1, self.target_q2):
            target.requires_grad_(False).eval()

    @torch.no_grad()
    def update_targets(self, tau):
        for target, online in ((self.target_q1, self.q1), (self.target_q2, self.q2)):
            for tp, op in zip(target.parameters(), online.parameters(), strict=True):
                tp.lerp_(op, tau)
            for tb, ob in zip(target.buffers(), online.buffers(), strict=True):
                tb.copy_(ob)
