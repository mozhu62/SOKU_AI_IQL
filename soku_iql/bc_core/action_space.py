from __future__ import annotations

from functools import lru_cache

import numpy as np


ACTION_SCHEMA = "soku_controller_joint144_v1"
ACTION_COUNT = 144
START_ACTION_ID = 144
PREVIOUS_ACTION_VOCAB = 145
NEUTRAL_ACTION_ID = 64
COMBAT_BUTTONS = ("melee", "dash", "light_projectile", "heavy_projectile")
RAW_ACTION_SCHEMA = "soku_controller_joint432_v1"
RAW_CARD_PROJECTION = "ignore_card_buttons_v1"


def _integer(value, name, low, high):
    array = np.asarray(value)
    if (not np.issubdtype(array.dtype, np.integer) or
            np.any((array < low) | (array > high))):
        raise ValueError(f"action schema 不兼容：{name} 必须是 {low}..{high} 的整数")
    return array.astype(np.int64, copy=False)


def encode(direction, combat_mask):
    """一个 ID 对应单游戏帧的方向和四个战斗按钮状态（不包含卡牌），方向始终是屏幕绝对九宫格。"""
    direction = _integer(direction, "direction", 1, 9)
    combat = _integer(combat_mask, "combat_mask", 0, 15)
    result = (direction - 1) * 16 + combat
    return int(result) if result.ndim == 0 else result


def decode(joint_action_id):
    value = _integer(joint_action_id, "joint_action_id", 0, ACTION_COUNT - 1)
    result = (value // 16 + 1, value % 16)
    return tuple(int(x) for x in result) if value.ndim == 0 else result


def validate_buttons(buttons, context=""):
    values = np.asarray(buttons)
    if values.ndim < 1 or values.shape[-1] != 4 or not np.isin(values, [0, 1]).all():
        raise ValueError(f"action schema 不兼容：{context} 战斗按钮必须是 A/D/B/C 四列 0/1")
    return values.astype(np.int64, copy=False)


def project_raw_buttons(buttons, context=""):
    """仅在旧 NPZ 导入边界投影六列按键；保留所有帧，不回写源文件。"""
    values = np.asarray(buttons)
    if values.ndim < 1 or values.shape[-1] != 6 or not np.isin(values, [0, 1]).all():
        raise ValueError(f"原始数据不兼容：{context} v4 按钮必须是六列 0/1")
    ignored = np.any(values[..., 4:] != 0, axis=-1)
    return values[..., :4].astype(np.int64), ignored


def from_controller(direction, buttons, context=""):
    values = validate_buttons(buttons, context)
    # 固定 bit 顺序 A、D、B、C；推理接口不再接收或产生卡牌命令。
    combat = (values * np.asarray([1, 2, 4, 8])).sum(-1)
    return encode(direction, combat)


def from_raw_axes(horizontal, vertical, buttons, positive_down=True, context=""):
    h = _integer(horizontal, "action_horizontal", -1, 1)
    v = _integer(vertical, "action_vertical", -1, 1)
    direction = 5 + h - 3 * v * (1 if positive_down else -1)
    return from_controller(direction, buttons, context)


def to_controller(joint_action_id):
    direction, combat = decode(joint_action_id)
    combat = np.asarray(combat)
    buttons = np.stack([(combat >> bit) & 1 for bit in range(4)], -1)
    return direction, buttons.astype(np.int64)


def action_name(joint_action_id):
    direction, combat = decode(joint_action_id)
    # 展示顺序 D+A+B+C 不改变存储 bit 顺序，6+D+A 表示同帧同时按下。
    names = [name for bit, name in ((1, "D"), (0, "A"), (2, "B"), (3, "C")) if combat & (1 << bit)]
    return "+".join([str(direction), *names])


@lru_cache(maxsize=1)
def _action_catalog():
    return tuple(action_name(i) for i in range(ACTION_COUNT))


def action_catalog():
    return list(_action_catalog())


def frequency_rows(counts, limit=20):
    counts = np.asarray(counts, dtype=np.int64)
    total = int(counts.sum())
    order = np.argsort(-counts, kind="stable")[:limit]
    return [{"joint_action_id": int(i), "action": _action_catalog()[i], "count": int(counts[i]),
             "fraction": int(counts[i]) / total if total else None} for i in order if counts[i] > 0]


def previous_actions(joint, duration, episode, valid, terminated):
    """在完整分片上先右移一帧，再取训练序列；绝不将 action_t 回灌到 observation_t。"""
    joint = _integer(joint, "joint_action_id", 0, ACTION_COUNT - 1)
    duration = _integer(duration, "action_duration", 1, np.iinfo(np.int64).max)
    count = len(joint)
    if any(np.asarray(x).shape != (count,) for x in (duration, episode, valid, terminated)):
        raise ValueError("上一帧动作历史数组长度不一致")
    previous = np.full(count, START_ACTION_ID, np.int64)
    age = np.zeros((count, 1), np.float32)
    connected = ((episode[1:] == episode[:-1]) & valid[:-1] & ~terminated[:-1])
    positions = np.flatnonzero(connected) + 1
    previous[positions] = joint[positions - 1]
    # duration 仅描述水平/垂直组合的持续帧数，按钮变化不重置该计数。
    age[positions, 0] = np.minimum(duration[positions - 1], 60) / 60.0
    return previous, age

