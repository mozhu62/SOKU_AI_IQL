from __future__ import annotations

import copy

import torch
from torch import nn

from .bc_core.models import BCNetwork


def expand_tcn_weights(network, source):
    """只允许向更长窗口追加完整卷积块，旧层键和形状必须全部匹配。"""
    destination = network.state_dict()
    blocks = {int(k.split('.')[2]) for k in source if k.startswith('tcn.blocks.')}
    old_count, new_count = len(blocks), len(network.tcn.blocks)
    if old_count not in (4, 5) or blocks != set(range(old_count)) or new_count <= old_count:
        raise ValueError("仅允许 TCN32/64 向更长窗口扩展，不允许降级或缺块")
    added = set(destination) - set(source)
    expected = {k for k in destination if any(k.startswith(f'tcn.blocks.{i}.') for i in range(old_count, new_count))}
    if set(source) - set(destination) or added != expected:
        raise ValueError("迁移只允许新增末尾 TCN 块，不能缺少或改变原网络的层")
    for key, value in source.items():
        if destination[key].shape != value.shape:
            raise ValueError(f"迁移权重形状不匹配：{key}")
        destination[key] = value
    network.load_state_dict(destination, strict=True)


def expand_tcn32_weights(network, source):
    # 保留上一版离线迁移脚本的导入入口。
    expand_tcn_weights(network, source)


class IQLNetworks(nn.Module):
    def __init__(self, model_config, actor_state=None):
        super().__init__()
        self.actor = BCNetwork(model_config)
        if actor_state is not None:
            if set(actor_state) != set(self.actor.state_dict()):
                expand_tcn_weights(self.actor, actor_state)
            else:
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
