from __future__ import annotations

from collections import deque

import numpy as np
import torch
from .streaming_tcn import StreamingTCN


class TCNObservationWindow:
    """真实观测驱动的窗口，与模型是否出动作、按键是否发送成功无关。"""

    size = 32

    def __init__(self, size=32, streaming=True):
        if size not in (32, 64, 256):
            raise ValueError("只支持真实32/64/256帧窗口")
        self.size = size
        self.rows = deque(maxlen=self.size)
        self.features = {}
        self.previous_action = self.previous_duration = None
        self.streaming = streaming
        self.stream = self.stream_key = self.stream_model = self.temporal = None
        self.processed_frames = 0
        self.rebuilt = False
        self.graph = None

    def reset(self):
        if self.graph is not None:
            self.graph.reset()
        self.rows.clear()
        self.features.clear()
        self.previous_action = self.previous_duration = None
        self.stream = self.stream_key = self.stream_model = self.temporal = None
        self.processed_frames = 0
        self.rebuilt = False

    def __len__(self):
        return len(self.rows)

    @property
    def key(self):
        return self.rows[-1][0] if self.rows else None

    def append(self, key, observation):
        if key == self.key:
            return
        if self.key is not None and (key[:2] != self.key[:2] or key[2] != self.key[2] + 1):
            raise ValueError("TCN 观测不连续；必须先重置片段，禁止伪造完整历史")
        self.rows.append((key, observation))
        keys = {frame_key for frame_key, _ in self.rows}
        self.features = {frame_key: feature for frame_key, feature in self.features.items() if frame_key in keys}
        self.previous_action = int(observation["previous_joint_action_id"])
        self.previous_duration = float(observation["previous_action_duration"][0])

    @torch.inference_mode()
    def logits(self, model, device, timing=None, tcn_device=None):
        tcn_device = device if tcn_device is None else tcn_device
        hybrid = torch.device(tcn_device) != torch.device(device)
        if len(self) != self.size:
            raise ValueError(f"真实连续历史不足 {self.size} 帧（当前 {len(self)}），本次不推理发键")
        if self.stream_model is not model:
            self.features.clear()
            self.stream = self.stream_key = self.temporal = None
            self.stream_model = model
        pending = [(key, obs) for key, obs in self.rows if key not in self.features]
        if timing: timing.mark('cache_lookup')
        if pending:
            # 只缓存每帧 228D 状态拼接；对象分支只编码当前帧，历史对象不会混进 TCN。
            batch = {name: torch.from_numpy(np.stack([obs[name] for _, obs in pending])).to(device)
                     for name in pending[0][1] if '_object_' not in name}
            if timing: timing.mark('history_pack_transfer')
            encoded = model.state_features(batch)
            if timing: timing.mark('state_features')
            if encoded.shape != (len(pending), model.memory_dim) or not torch.isfinite(encoded).all():
                raise ValueError("TCN 历史特征形状不匹配或含 NaN/Inf")
            self.features.update({key: feature for (key, _), feature in zip(pending, encoded.unbind(0))})
            if timing: timing.mark('feature_check_wait')
        self.rebuilt = False
        if self.streaming:
            first = self.rows[0][0]
            if self.stream_key is None or self.stream_key[:2] != first[:2] or self.stream_key[2] < first[2]-1:
                # 暂停期间未计算的帧已出窗口时，重建最近完整窗口；不能跳过中间帧继续缓存。
                self.stream = self.graph if self.graph is not None else StreamingTCN(model.tcn)
                if self.graph is not None:
                    self.graph.reset()
                self.stream_key = None
                self.rebuilt = True
            fresh = [key for key, _ in self.rows if self.stream_key is None or key[2] > self.stream_key[2]]
            self.processed_frames = len(fresh)
            if fresh:
                chunk = torch.stack([self.features[key] for key in fresh])[None]
                if hybrid:
                    chunk = chunk.to(tcn_device)
                    if timing: timing.mark('tcn_input_transfer')
                self.temporal = self.stream.forward(chunk)[:, -1].detach().clone()
                self.stream_key = fresh[-1]
            temporal = self.temporal
        else:
            history = torch.stack([self.features[key] for key, _ in self.rows])[None]
            if hybrid:
                history = history.to(tcn_device)
                if timing: timing.mark('tcn_input_transfer')
            temporal = model.tcn(history)[:, -1]
            self.processed_frames = self.size
        if timing: timing.mark('tcn')
        current = model.current_encoder.forward_features(self.features[self.key][None])
        if timing: timing.mark('current_encoder')
        latest = {name: torch.from_numpy(value).unsqueeze(0).to(device)
                  for name, value in self.rows[-1][1].items() if '_object_' in name}
        if timing: timing.mark('current_pack_transfer')
        objects = model.encode_objects(latest)
        if timing: timing.mark('objects')
        if hybrid:
            # 在 CPU 分支完成后再等待 GPU，只回传最后一个 256D 特征。
            temporal = temporal.to(device=device, dtype=current.dtype)
            if timing: timing.mark('tcn_return_wait')
        result = model._fuse(current, temporal, objects)
        if timing: timing.mark('fusion_head')
        return result
