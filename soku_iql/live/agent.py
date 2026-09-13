from __future__ import annotations

import hashlib
import math
import time
from pathlib import Path

import torch

from .policy_checkpoint import load
from ..bc_core.models import BCNetwork
from ..bc_core.action_space import ACTION_COUNT, to_controller, decode, action_name
from .observation import ObservationBuilder
from .tcn_window import TCNObservationWindow
from .batch_graph_candidate import BatchGraphCandidate


class LiveAgent:
    def __init__(self, path: Path, config):
        if not path.is_file():
            raise FileNotFoundError(f"找不到实战模型：{path}；请从训练服务器复制完整的 .pt 文件")
        before = path.stat()
        torch.set_num_threads(config["cpu_threads"])
        requested = config["device"]
        self.hybrid = requested == 'hybrid'
        if self.hybrid and not torch.cuda.is_available():
            raise ValueError('hybrid 需要可用 CUDA，请检查环境或显式选择 cpu')
        requested = 'cpu' if self.hybrid else requested
        self.device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if requested == "auto" else requested)
        self.tcn_device = torch.device('cuda:0') if self.hybrid else self.device
        self.device_label = 'hybrid (CPU + TCN CUDA:0)' if self.hybrid else str(self.device)
        package = load(path)
        self.trained_algorithm = package.get("provenance", {}).get("trained_algorithm", "bc")
        self.model = BCNetwork(package["spec"]["model"], package["network_version"])
        if self.model.spec != package["spec"]:
            raise ValueError("模型输入/Joint Action schema 不兼容；只接受新版 144-way IQL/BC 策略")
        # BC 包的权重键是 model；不能把 CQL 的 online Q 网络或优化器当作策略加载。
        self.model.load_state_dict(package["model"], strict=True)
        self.model.to(self.device).eval().requires_grad_(False)
        # 混合模式只将 TCN 放 GPU，编码和输出仍为 CPU FP32。
        if self.hybrid:
            self.model.tcn.to(self.tcn_device)
        self.inference_amp = config.get('amp', False) and self.tcn_device.type == 'cuda'
        self.inference_dtype = ('CPU FP32 / TCN ' if self.hybrid else '') + ('AMP FP16' if self.inference_amp else 'FP32')
        self.temporal_mode = self.model.temporal_mode
        if config["environment"]["decision_interval_frames"] != 1:
            raise ValueError("TCN 按连续游戏帧训练，实战 decision_interval_frames 必须为 1，不能用稀疏决策冒充连续帧")
        # 回读方向使用模型训练时的轴约定；Joint Action 的方向已经是屏幕绝对九宫格。
        self.vertical_positive_is_down = bool(package["config"]["data"]["vertical_positive_is_down"])
        self.builder = ObservationBuilder(package["normalization"], config["environment"]["player_side"],
                                          self.vertical_positive_is_down)
        self.step = int(package["step"])
        self.action_selection = "deterministic_argmax_logits"
        self.path = str(path)
        with path.open("rb") as stream:
            self.sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        after = path.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
            raise ValueError("加载过程中模型文件发生变化，请先复制为固定文件再开始评估")
        self.memory = None
        self.tcn_window = TCNObservationWindow(self.model.tcn.context_frames, config.get('streaming_tcn', True))
        if config.get('tcn_cuda_graph', True) and self.tcn_window.streaming and self.tcn_device.type == 'cuda':
            self.tcn_window.graph = BatchGraphCandidate(self.model.tcn, self.tcn_device, self.inference_amp)

    def reset(self):
        self.memory = None
        self.builder.reset()
        self.tcn_window.reset()

    def observe_tcn_frame(self, payload, resources):
        key = (int(payload.gameProcessId), int(payload.currentRound), int(payload.battleFrame))
        if key != self.tcn_window.key:
            self.tcn_window.append(key, self.builder.build(payload, resources))

    @torch.inference_mode()
    def predict(self, payload, resources=None):
        started = time.perf_counter()
        key = (int(payload.gameProcessId), int(payload.currentRound), int(payload.battleFrame))
        if self.tcn_window.key != key:
            raise ValueError("推理帧不对应 TCN 窗口末帧，禁止使用错位历史")
        with torch.autocast(device_type=self.tcn_device.type, dtype=torch.float16, enabled=self.inference_amp):
            logits = self.tcn_window.logits(self.model, self.device, tcn_device=self.tcn_device)
        memory = None
        previous_id, previous_duration = self.tcn_window.previous_action, self.tcn_window.previous_duration
        if (logits.shape != (1, ACTION_COUNT) or not torch.isfinite(logits).all()
                or (memory is not None and not torch.isfinite(memory).all())):
            raise ValueError("策略输出形状不是 [1,144] 或 logits/时序状态含 NaN/Inf，已停止控制")
        # 转 CPU 同步 CUDA，使耗时覆盖实际计算，而不是只测异步提交。
        joint_logits = logits[0].float().cpu()
        probabilities = joint_logits.softmax(-1)
        # 固定权重评估采用 argmax；softmax 仅用于显示，不采样、不加噪声或动作保护。
        action = int(joint_logits.argmax())
        entropy = -(probabilities * joint_logits.log_softmax(-1)).sum()
        if not torch.isfinite(probabilities).all() or not torch.isfinite(entropy):
            raise ValueError("策略分类概率或熵含 NaN/Inf，已停止控制")
        direction, buttons = to_controller(action)
        _, combat = decode(action)
        return {"joint_action_id": action, "action": action_name(action), "combat_mask": combat,
                "direction": direction, "buttons": tuple(int(x) for x in buttons),
                "joint_logits": joint_logits.tolist(), "joint_probabilities": probabilities.tolist(),
                "selected_probability": float(probabilities[action]), "entropy": float(entropy),
                "normalized_entropy": float(entropy) / math.log(ACTION_COUNT),
                "output_semantics": "categorical_logits", "action_selection": self.action_selection,
                "temporal_mode": self.temporal_mode,
                "context_frames_used": len(self.tcn_window),
                "tcn_computed_frames": self.tcn_window.processed_frames,
                "tcn_cache_rebuilt": self.tcn_window.rebuilt,
                "tcn_backend": self.tcn_window.graph.last_path if self.tcn_window.graph is not None else 'eager',
                "context_first_frame": self.tcn_window.rows[0][0][2],
                "observation_frame": int(payload.battleFrame), "observation_round": int(payload.currentRound),
                "sample_serial": int(payload.sampleSerial),
                "previous_joint_action_id": previous_id,
                "previous_action_duration": previous_duration,
                "inference_ms": (time.perf_counter() - started) * 1000,
                "resource_inputs": self.builder.resource_summary}, memory
