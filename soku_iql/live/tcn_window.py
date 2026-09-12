from __future__ import annotations

from collections import deque

import numpy as np
import torch


class TCNObservationWindow:
    """真实观测驱动的窗口，与模型是否出动作、按键是否发送成功无关。"""

    size = 32

    def __init__(self):
        self.rows = deque(maxlen=self.size)
        self.features = {}
        self.previous_action = self.previous_duration = None

    def reset(self):
        self.rows.clear()
        self.features.clear()
        self.previous_action = self.previous_duration = None

    def __len__(self):
        return len(self.rows)

    @property
    def key(self):
        return self.rows[-1][0] if self.rows else None

    def append(self, key, observation):
        if key == self.key:
            return
        if self.key is not None and (key[:2] != self.key[:2] or key[2] != self.key[2] + 1):
            raise ValueError("TCN 观测不连续；必须先重置片段，禁止重复帧或稀疏帧冒充 32 帧")
        self.rows.append((key, observation))
        keys = {frame_key for frame_key, _ in self.rows}
        self.features = {frame_key: feature for frame_key, feature in self.features.items() if frame_key in keys}
        self.previous_action = int(observation["previous_joint_action_id"])
        self.previous_duration = float(observation["previous_action_duration"][0])

    @torch.inference_mode()
    def logits(self, model, device):
        if len(self) != self.size:
            raise ValueError(f"真实连续历史不足 {self.size} 帧（当前 {len(self)}），本次不推理发键")
        pending = [(key, obs) for key, obs in self.rows if key not in self.features]
        if pending:
            # 只缓存每帧 228D 状态拼接；对象分支只编码当前帧，历史对象不会混进 TCN。
            batch = {name: torch.from_numpy(np.stack([obs[name] for _, obs in pending])).to(device)
                     for name in pending[0][1]}
            encoded = model.state_features(batch)
            if encoded.shape != (len(pending), model.memory_dim) or not torch.isfinite(encoded).all():
                raise ValueError("TCN 历史特征形状不匹配或含 NaN/Inf")
            self.features.update({key: feature for (key, _), feature in zip(pending, encoded.unbind(0))})
        history = torch.stack([self.features[key] for key, _ in self.rows])[None]
        temporal = model.tcn(history)[:, -1]
        current = model.current_encoder.forward_features(history[:, -1])
        latest = {name: torch.from_numpy(value).unsqueeze(0).to(device)
                  for name, value in self.rows[-1][1].items()}
        return model._fuse(current, temporal, model.encode_objects(latest))
