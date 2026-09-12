from __future__ import annotations

import numpy as np


RESOURCE_SUFFIXES = ("skill_valid_mask", "skill_variants")
IGNORED_RESOURCE_SUFFIXES = (
    "skill_levels", "skill_effective_levels", "card_state",
    "hand_card_ids", "hand_card_costs", "hand_mask",
)


def validate_player_resources(raw: dict, shape: tuple, label: str) -> None:
    """只验证本实验使用的技能类型；不要求读取已移除的卡牌与等级。"""
    for key, tail in (("skill_valid_mask", ()), ("skill_variants", (4,))):
        value = raw.get(key)
        if (value is None or value.shape != (*shape, *tail)
                or not np.issubdtype(value.dtype, np.integer)):
            raise ValueError(f"{label}.{key} 缺失、形状错误或不是整数")
    valid = raw["skill_valid_mask"]
    if ((valid < 0) | (valid > 15)).any():
        raise ValueError(f"{label} 技能有效位超出四个槽")
    mask = (valid[..., None] & (1 << np.arange(4))) != 0
    variants = raw["skill_variants"]
    if (((variants < -1) | (variants > 2)).any()
            or not np.array_equal(variants >= 0, mask)
            or (variants[~mask] != -1).any()):
        raise ValueError(f"{label} 技能类型与有效 mask 不一致")


def player_resource_observation(raw: dict) -> dict[str, np.ndarray]:
    mask = (raw["skill_valid_mask"][..., None].astype(np.int64) & (1 << np.arange(4))) != 0
    # 每槽仅一个 variant token：0 为未知，真实类型 0/1/2 编为 1/2/3。
    return {"skill_categorical": np.where(mask, raw["skill_variants"].astype(np.int64) + 1, 0),
            "skill_mask": mask}


def resource_observation(shard: dict, indices) -> dict[str, np.ndarray]:
    result = {}
    for side in ("self", "opponent"):
        raw = {key: shard[f"{side}_{key}"][indices] for key in RESOURCE_SUFFIXES}
        result.update({f"{side}_{key}": value for key, value in player_resource_observation(raw).items()})
    return result


def normalization_arrays(normalization):
    result = {}
    for group, size in (("state", 18), ("objects", 8), ("optional_state", 4)):
        raw = normalization.get(group, {})
        mean, std = (np.asarray(raw.get(key), np.float32) for key in ("mean", "std"))
        if (mean.shape != (size,) or std.shape != (size,) or not np.isfinite(mean).all()
                or not np.isfinite(std).all() or (std <= 0).any()):
            raise ValueError(f"归一化 schema 不兼容：{group} 缺失或无效；旧 checkpoint 不可用于 joint144")
        result[group] = mean, std
    counts = np.asarray(normalization["optional_state"].get("counts"))
    if counts.shape != (4,) or not np.issubdtype(counts.dtype, np.integer) or (counts < 0).any():
        raise ValueError("optional_state 有效样本数缺失或无效")
    return result

