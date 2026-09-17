"""诊断配置与 Joint144 按键类别；不参与训练目标。"""
import copy
import math

from .bc_core.action_space import ACTION_COUNT, decode


DEFAULTS = {
    "enabled": True, "train_interval": 100, "probe_interval": 1000,
    "adv_near_zero_threshold": 0.05,
    "probe_seed": 271828, "probe_batches": 4, "probe_batch_size": 4,
    "probe_sequence_length": 32, "probe_replays_per_batch": 4,
    "probe_max_samples": 128, "near_distance": 100.0, "far_distance": 300.0,
    "action_classes": {},
    "warnings": {
        "weight_clip_ratio": 0.05, "q_gap_mean": 10.0, "q_gap_p95": 20.0,
        "q_std": 50.0, "adv_std_min": 0.01, "near_zero_ratio": 0.90,
        "class_weight_share": 0.50, "class_share_amplification": 2.0,
        "class_min_samples": 20, "attack_ratio_excess": 0.15,
        "switch_ratio_excess": 0.15, "attack_adv_gap": 0.10,
        "beta_adv_std": 2.0,
    },
}


def class_map(settings):
    # 多按钮可以同时按下；混合攻击单列，避免强行归到 Melee 或 Projectile。
    names = []
    for action in range(ACTION_COUNT):
        direction, buttons = decode(action)
        melee, projectile = bool(buttons & 1), bool(buttons & 12)
        names.append("MixedAttack" if melee and projectile else "Melee" if melee else
                     "Projectile" if projectile else "Dash" if buttons & 2 else
                     "Neutral" if direction == 5 else "Movement")
    seen = set()
    for name, ids in settings.get("action_classes", {}).items():
        if not isinstance(name, str) or not name or not isinstance(ids, list):
            raise ValueError("diagnostics.action_classes 必须是类别名到动作 ID 列表的映射")
        for action in ids:
            if type(action) is not int or not 0 <= action < ACTION_COUNT or action in seen:
                raise ValueError("诊断类别的动作 ID 必须在 0..143 内且不得重复")
            seen.add(action)
            names[action] = name
    return names


def settings(supplied=None):
    supplied = {} if supplied is None else supplied
    if not isinstance(supplied, dict) or set(supplied) - set(DEFAULTS):
        raise ValueError("diagnostics 包含未知配置项")
    result = copy.deepcopy(DEFAULTS)
    result.update(supplied)
    limits = supplied.get("warnings", {})
    if not isinstance(limits, dict) or set(limits) - set(DEFAULTS["warnings"]):
        raise ValueError("diagnostics.warnings 包含未知阈值")
    result["warnings"] = {**DEFAULTS["warnings"], **limits}
    if type(result["enabled"]) is not bool or not isinstance(result["action_classes"], dict):
        raise ValueError("诊断 enabled 必须为布尔值，action_classes 必须为映射")
    for key in ("train_interval", "probe_interval", "probe_batches", "probe_batch_size",
                "probe_sequence_length", "probe_replays_per_batch", "probe_max_samples"):
        if type(result[key]) is not int or result[key] < 1:
            raise ValueError(f"diagnostics.{key} 必须为正整数")
    if type(result["probe_seed"]) is not int or result["probe_seed"] < 0:
        raise ValueError("probe_seed 必须是非负整数")
    for key in ("adv_near_zero_threshold", "near_distance", "far_distance"):
        if type(result[key]) not in (int, float) or not math.isfinite(result[key]) or result[key] < 0:
            raise ValueError(f"diagnostics.{key} 必须是非负有限数")
    if result["far_distance"] <= result["near_distance"]:
        raise ValueError("far_distance 必须大于 near_distance")
    for key, value in result["warnings"].items():
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"诊断告警阈值 {key} 必须是非负有限数")
    class_map(result)
    return result
