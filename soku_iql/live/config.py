from __future__ import annotations

import copy
import math
import re

import yaml

from ..config import resolve
from ..bc_core.schema import BUTTON_FEATURES
from .windows_api import parse_virtual_key


DEFAULTS = {
    "checkpoint": "outputs/iql_suika/actor_bc.pt",
    "device": "cpu", "cpu_threads": 2,
    "amp": False, "streaming_tcn": True, "tcn_cuda_graph": True,
    "output": "outputs/evaluations", "rounds": 20,
    "environment": {
        "player_side": "left",
        "active_match_states": [2],
        "cpu_difficulty_label": "未确认，请按游戏设置填写",
        "decision_interval_frames": 1, "max_inference_lag_frames": 6,
        "max_memory_gap_frames": 6, "max_snapshot_age_ms": 250,
        "stale_timeout_seconds": 1.0, "focus_delay_seconds": 0.3,
        "watchdog_seconds": 1.0, "poll_seconds": 0.002,
    },
    "keyboard": {"up": "W", "down": "S", "left": "A", "right": "D",
                 "melee": "J", "dash": "K", "light_projectile": "I", "heavy_projectile": "L"},
    "restart": {"enabled": True, "confirm_key": "Z", "hold_seconds": 0.05,
                "interval_seconds": 0.25, "timeout_seconds": 60.0, "max_presses": 240},
    "web": {"host": "127.0.0.1", "port": 8826, "port_attempts": 10},
}


def load_config(path="configs/live_eval.yaml"):
    config = copy.deepcopy(DEFAULTS)
    with resolve(path).open(encoding="utf-8") as stream:
        supplied = yaml.safe_load(stream) or {}
    if not isinstance(supplied, dict):
        raise ValueError("实战配置必须是 YAML 映射")
    for key, value in supplied.items():
        if key == "environment" and isinstance(value, dict):
            # 旧 YAML 的模式/角色固定值已弃用；兼容忽略，不重新引入角色拦截。
            value = {name: item for name, item in value.items()
                     if name not in ("battle_mode", "battle_submode", "self_character", "opponent_character")}
        if key not in config:
            raise ValueError(f"未知实战配置项：{key}")
        if isinstance(config[key], dict):
            if not isinstance(value, dict) or set(value) - set(config[key]):
                raise ValueError(f"实战配置 {key} 包含未知字段或格式错误")
            config[key].update(value)
        else:
            config[key] = value
    validate(config)
    return config


def validate(config):
    for name, default in (('amp', False), ('streaming_tcn', True), ('tcn_cuda_graph', True)):
        if type(config.get(name, default)) is not bool:
            raise ValueError(f'{name} 必须是布尔值')
    def integer(value, low, high):
        return type(value) is int and low <= value <= high

    env, restart = config["environment"], config["restart"]
    if any(not isinstance(config[key], str) or not config[key].strip() for key in ("checkpoint", "output")):
        raise ValueError("模型路径和评估输出目录必须是非空字符串")
    if not isinstance(config["device"], str) or not re.fullmatch(r"auto|cpu|hybrid|cuda(?::[0-9]+)?", config["device"]):
        raise ValueError("推理设备应为 cpu / cuda / cuda:0 / auto / hybrid")
    if not integer(config["rounds"], 0, 10000) or not integer(config["cpu_threads"], 1, 64):
        raise ValueError("rounds 应为 0～10000（0 为不限局数），cpu_threads 应为 1～64")
    if env["player_side"] not in ("left", "right"):
        raise ValueError("player_side 必须是 left（1P）或 right（2P）")
    if env["decision_interval_frames"] != 1:
        raise ValueError("TCN 实战要求 decision_interval_frames=1，以便逐帧维护模型所需的真实历史窗口")
    for key in ("max_inference_lag_frames", "max_memory_gap_frames"):
        if not integer(env[key], 1, 120):
            raise ValueError(f"{key} 必须是 1～120 帧")
    for key in ("max_snapshot_age_ms", "stale_timeout_seconds", "watchdog_seconds", "poll_seconds"):
        if not isinstance(env[key], (int, float)) or not math.isfinite(env[key]) or env[key] <= 0:
            raise ValueError(f"{key} 必须是有限正数")
    if not 0 <= env["focus_delay_seconds"] <= 10 or not 0.001 <= env["poll_seconds"] <= 0.1:
        raise ValueError("聚焦延迟或采样间隔超出范围")
    if not env["active_match_states"] or not all(type(x) is int for x in env["active_match_states"]):
        raise ValueError("active_match_states 必须是非空整数列表")
    if type(restart["enabled"]) is not bool:
        raise ValueError("restart.enabled 必须是布尔值")
    if not (0 < restart["hold_seconds"] < restart["interval_seconds"] <= 10):
        raise ValueError("续局按键保持时间必须小于点按间隔")
    if not 1 <= restart["timeout_seconds"] <= 600 or not integer(restart["max_presses"], 1, 2000):
        raise ValueError("续局超时或点按次数不合法")
    if set(config["keyboard"]) != {"up", "down", "left", "right", *BUTTON_FEATURES}:
        raise ValueError("按键配置应包括四方向与四个按钮")
    # 在停止旧会话前验证键位；这里只解析键名，不打开 Windows API、不发送按键。
    try:
        keys = [parse_virtual_key(value) for value in config["keyboard"].values()]
        confirm = parse_virtual_key(restart["confirm_key"])
    except (TypeError, AttributeError, ValueError) as exc:
        raise ValueError(f"按键映射无效：{exc}") from exc
    if len(set(keys)) != len(keys) or 0x79 in keys or confirm == 0x79:
        raise ValueError("八个游戏按键必须互不重复；游戏键和续局键均不能占用 F10")
    if config["web"]["host"] != "127.0.0.1":
        raise ValueError("实战键盘控制工作台仅监听 127.0.0.1；离线训练服务的局域网设置不受影响")
    if not integer(config["web"]["port"], 1024, 65535) or not integer(config["web"]["port_attempts"], 1, 100):
        raise ValueError("实战网页端口或端口重试次数无效")
