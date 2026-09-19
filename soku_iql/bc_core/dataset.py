from __future__ import annotations

import hashlib
import json
import logging
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np

from .config import resolve
from .schema import manifest, OPTIONAL_STATE_FEATURES
from .action_space import (
    ACTION_SCHEMA, ACTION_COUNT, NEUTRAL_ACTION_ID, RAW_ACTION_SCHEMA, RAW_CARD_PROJECTION,
    project_raw_buttons, from_raw_axes, previous_actions,
)
from .resources import (
    RESOURCE_SUFFIXES, IGNORED_RESOURCE_SUFFIXES, resource_observation, validate_player_resources, normalization_arrays,
)
from .storage import atomic_json
from .action_diagnostics import DIAGNOSTIC_VERSION, dataset_action_counts, merge_dataset_counts
from .weather import compress_weather


LOGGER = logging.getLogger(__name__)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def split_replays(config, progress, cancelled=lambda: False):
    root, path = resolve(config["data"]["directory"]), resolve(config["data"]["split_file"])
    files = sorted(root.rglob("*.npz"))
    if len(files) < 2:
        raise ValueError("至少需要两份完整 NPZ 才能划分训练/验证集；请先完成 REP 转换")
    entries = {}
    for index, file in enumerate(files):
        if cancelled():
            raise InterruptedError("数据核对已取消")
        progress(f"核对分片 {index + 1}/{len(files)}：{file.name}")
        with file.open("rb") as stream:
            sha = hashlib.file_digest(stream, "sha256").hexdigest()
        entries[file.relative_to(root).as_posix()] = sha
    if path.exists():
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("files") != entries or result.get("seed") != config["seed"]:
            raise ValueError("数据或 seed 与固定划分不一致；请为新数据指定新的 split_file 和输出目录")
    else:
        # 完全相同的 NPZ 按内容哈希归为一组，防止复制文件混入另一集合。
        groups = {}
        for name, sha in entries.items():
            groups.setdefault(sha, []).append(name)
        keys = sorted(groups)
        if len(keys) < 2:
            raise ValueError("数据只有一份独立内容，无法隔离训练和验证")
        np.random.default_rng(config["seed"]).shuffle(keys)
        count = min(len(keys) - 1, max(1, int(len(keys) * 0.8)))
        result = {"version": 1, "seed": config["seed"], "train_fraction": 0.8, "files": entries,
                  "train": sorted(name for key in keys[:count] for name in groups[key]),
                  "validation": sorted(name for key in keys[count:] for name in groups[key])}
        result["sha256"] = digest(result)
        atomic_json(path, result)
    payload = {key: value for key, value in result.items() if key != "sha256"}
    if result.get("sha256") != digest(payload):
        raise ValueError("划分清单完整性校验失败")
    train, val = set(result["train"]), set(result["validation"])
    if not train or not val or train & val or train | val != set(entries):
        raise ValueError("划分存在重复、遗漏或空集合")
    if {entries[x] for x in train} & {entries[x] for x in val}:
        raise ValueError("训练集与验证集包含相同分片内容")
    return result


class Moments:
    def __init__(self, size):
        self.count = 0
        self.mean = np.zeros(size, np.float64)
        self.m2 = np.zeros(size, np.float64)

    def update(self, values):
        if not len(values):
            return
        if not np.isfinite(values).all():
            raise ValueError("归一化数据含 NaN/Inf")
        mean = values.mean(0, dtype=np.float64)
        variance = values.var(0, dtype=np.float64)
        delta = mean - self.mean
        total = self.count + len(values)
        self.m2 += variance * len(values) + delta ** 2 * self.count * len(values) / total
        self.mean += delta * len(values) / total
        self.count = total

    def export(self):
        std = np.sqrt(self.m2 / max(1, self.count))
        return {"mean": self.mean.tolist(), "std": np.where(std < 1e-6, 1.0, std).tolist(), "count": self.count}


def read_shard(path: Path, positive_down: bool, keep_terminal: bool = False):
    try:
        with np.load(path, allow_pickle=False) as source:
            meta = json.loads(str(source["metadata_json"].item()))
            for key, expected in manifest().items():
                if meta.get(key) != expected:
                    raise ValueError(f"字段清单不兼容：{key}，请用当前采集预处理脚本重新生成")
            if meta.get("continuous_normalized") is not False:
                raise ValueError("分片必须保存未归一化原值")
            # BC 不读取奖励；终局标记仅用于划分连续历史，不构造回报或下一状态目标。
            ignored = {"metadata_json", "rewards"} | {
                f"{side}_{suffix}" for side in ("self", "opponent") for suffix in IGNORED_RESOURCE_SUFFIXES
            }
            shard = {key: source[key] for key in source.files if key not in ignored}
        count = len(shard["episode_id"])
        if meta.get("action_schema", RAW_ACTION_SCHEMA) != RAW_ACTION_SCHEMA:
            raise ValueError("action schema 不兼容：分片声明了不同的动作语义")
        if meta.get("action_shift") not in (0, 1):
            raise ValueError("缺少可靠的 action_shift，无法对齐实际输入历史")
        shard["action_shift"] = np.asarray(meta["action_shift"], np.int8)
        optional = ("state_optional_continuous", "state_optional_mask")
        if any(key in shard for key in optional):
            if meta.get("optional_state_continuous") != list(OPTIONAL_STATE_FEATURES):
                raise ValueError("可选人物状态字段顺序不兼容")
            if any(key not in shard or shard[key].shape != (count, 4) for key in optional):
                raise ValueError("可选人物状态与有效 mask 必须同时存在且为 [frames,4]")
            if not np.isin(shard[optional[1]], [0, 1]).all():
                raise ValueError("可选人物状态 mask 必须为 0/1")
            if not np.isfinite(shard[optional[0]][shard[optional[1]].astype(bool)]).all():
                raise ValueError("有效 max_spirit/hitstop 中存在 NaN/Inf")
        else:
            # 旧资源 NPZ 并未存这些值；显式无效，不把零伪装成角色真实状态。
            shard[optional[0]] = np.zeros((count, 4), np.float32)
            shard[optional[1]] = np.zeros((count, 4), bool)
        expected = {"state_continuous": (count, 18), "state_categorical": (count, 5),
                    "tactical_state": (count, 9), "action_buttons": (count, 6)}
        expected.update({key: (count,) for key in ("episode_id", "action_horizontal", "action_vertical",
                                                  "action_duration", "transition_valid", "terminated")})
        for key, shape in expected.items():
            if shard[key].shape != shape or not np.isfinite(shard[key]).all():
                raise ValueError(f"{key} 形状错误或含非有限值")
        for side in ("self", "opponent"):
            validate_player_resources({key: shard.get(f"{side}_{key}") for key in RESOURCE_SUFFIXES}, (count,), side)
        for key in ("state_categorical", "episode_id"):
            if not np.issubdtype(shard[key].dtype, np.integer):
                raise ValueError(f"{key} 必须使用整数类型")
        for key in ("transition_valid", "terminated", "tactical_state"):
            if not np.isin(shard[key], [0, 1]).all():
                raise ValueError(f"{key} 必须使用 0/1 标志")
        if count < 2:
            raise ValueError("分片少于两帧")
        for key in ("action_horizontal", "action_vertical"):
            if not np.isin(shard[key], [-1, 0, 1]).all():
                raise ValueError(f"{key} 必须为 -1/0/1")
        if not np.isin(shard["action_buttons"], [0, 1]).all():
            raise ValueError("按钮必须为六个独立的 0/1 列")
        for side in ("self", "opponent"):
            numeric, categorical, offsets = (shard[f"{side}_object_{suffix}"] for suffix in
                                               ("numerical", "categorical", "offsets"))
            if (numeric.ndim != 2 or numeric.shape[1] != 8 or categorical.shape != (len(numeric), 2)
                    or offsets.shape != (count + 1,) or offsets[0] != 0 or offsets[-1] != len(numeric)
                    or not np.issubdtype(offsets.dtype, np.integer)
                    or not ((np.diff(offsets) >= 0) & (np.diff(offsets) <= 3)).all()
                    or not np.issubdtype(categorical.dtype, np.integer)
                    or not np.isfinite(numeric).all() or not np.isfinite(categorical).all()):
                raise ValueError(f"{side} 对象、offsets 或最近三对象约束错误")
        # 先去掉卡牌两列，再构造标签及上一帧历史；卡牌独有切换自然成为 hold。
        raw_buttons = shard["action_buttons"]
        shard["action_buttons"], ignored_cards = project_raw_buttons(raw_buttons, path.name)
        shard["card_input_ignored_rows"] = np.asarray(np.count_nonzero(ignored_cards), np.int64)
        joint = from_raw_axes(shard["action_horizontal"], shard["action_vertical"],
                              shard["action_buttons"], positive_down, path.name)
        if "joint_action_id" in shard:
            stored = shard["joint_action_id"]
            # 既有 v4 Joint432 标签只能用于核对，训练标签始终从真实原始按键重新生成。
            cleaned_change = raw_buttons[:, 4] * (1 - raw_buttons[:, 5])
            expected_old = joint * 3 + cleaned_change + 2 * raw_buttons[:, 5]
            if not np.array_equal(stored, expected_old):
                raise ValueError("原始 Joint432 标签与原始 Controller State 不一致")
        shard["joint_action_id"] = joint
        valid = shard["transition_valid"].astype(bool).copy()
        valid[-1] = False
        valid[:-1] &= shard["episode_id"][:-1] == shard["episode_id"][1:]
        terminal = shard["terminated"].astype(bool)
        # 沿用 NPZ 的动作对齐规则；终局后的观测不能作为下一段专家动作起点。
        valid[1:] &= ~terminal[:-1]
        # 从完整专家标签及既有连续性规则生成监督元数据，而非从 observation 反取 PALR 标签。
        # 单独复制网络输入，防止后续输入侧变换影响真实上一帧专家监督。
        shard["previous_expert_action_id"], shard["previous_action_duration"] = previous_actions(
            joint, shard["action_duration"], shard["episode_id"], valid, terminal)
        shard["previous_joint_action_id"] = shard["previous_expert_action_id"].copy()
        starts = np.flatnonzero(valid & ~np.r_[False, valid[:-1]])
        ends = np.flatnonzero(valid & ~np.r_[valid[1:], False]) + 1
        shard["segments"] = np.column_stack((starts, ends))
        shard["segment_cumulative"] = np.cumsum(ends - starts)
        if not len(starts):
            raise ValueError("没有有效连续转移")
        for key in ("action_horizontal", "action_vertical", "action_duration", "action_buttons", "transition_valid", "episode_id"):
            shard.pop(key)
        if not keep_terminal:
            shard.pop("terminated")
        return shard
    except Exception as error:
        raise ValueError(f"分片读取失败 {path.name}：{error}") from error


class ReplayStore:
    """借鉴 DQN 的分片预加载，使用有容量上限的缓存，批量索引而非逐帧 Dataset 调用。"""

    def __init__(self, config, split, progress, cancelled=lambda: False, normalization=None):
        self.config, self.split = config, split
        self.root = resolve(config["data"]["directory"])
        self.budget = int(config["data"]["cache_gb"] * 1024 ** 3)
        self.cache, self.bytes, self.hits, self.misses = OrderedDict(), 0, 0, 0
        self.lock = threading.RLock()
        self.info = {}
        state_stats, object_stats = Moments(18), Moments(8)
        optional_stats = [Moments(1) for _ in range(4)]
        shifts = set()
        train_names = set(split["train"])
        for index, name in enumerate(split["validation"] + split["train"]):
            if cancelled():
                raise InterruptedError("数据准备已取消")
            progress(f"预读分片 {index + 1}/{len(split['files'])}：{name}")
            shard = self.get(name)
            shifts.add(int(shard["action_shift"]))
            valid = np.zeros(len(shard["state_continuous"]), dtype=bool)
            for start, end in shard["segments"]:
                valid[start:end] = True
            self.info[name] = {"name": name, "split": "train" if name in train_names else "validation",
                               "frames": len(valid), "transitions": int(valid.sum()),
                               "joint_counts": np.bincount(shard["joint_action_id"][valid], minlength=ACTION_COUNT).tolist()}
            # 复用正式标签的有效片段，一次预读时计数；不改采样、归一化或历史对齐。
            self.info[name]["action_history"] = dataset_action_counts(
                shard["joint_action_id"], shard["previous_joint_action_id"], valid)
            self.info[name]["card_input_ignored_rows"] = int(shard["card_input_ignored_rows"])
            self.info[name]["optional_state_counts"] = shard["state_optional_mask"][valid].sum(0).astype(int).tolist()
            self.info[name]["resource_coverage"] = {
                side: (((shard[f"{side}_skill_valid_mask"][valid, None].astype(np.int64)
                         & (1 << np.arange(4))) != 0).sum(0).tolist())
                for side in ("self", "opponent")
            }
            if normalization is None and name in train_names:
                state_stats.update(shard["state_continuous"])
                for column, stats in enumerate(optional_stats):
                    mask = shard["state_optional_mask"][:, column].astype(bool)
                    stats.update(shard["state_optional_continuous"][mask, column, None])
                for side in ("self", "opponent"):
                    object_stats.update(shard[f"{side}_object_numerical"])
        self.action_history = {
            group: merge_dataset_counts([self.info[name]["action_history"] for name in split[group]])
            for group in ("train", "validation")
        }
        if len(shifts) != 1:
            raise ValueError("数据集中 action_shift 不一致；不能混合不同状态/动作帧对齐约定")
        optional_export = {key: [stats.export()[key][0] for stats in optional_stats] for key in ("mean", "std")}
        optional_export["counts"] = [stats.count for stats in optional_stats]
        self.normalization = normalization or {"split_hash": split["sha256"], "state": state_stats.export(),
                                                "objects": object_stats.export(),
                                                "optional_state": optional_export,
                                                "action_schema": ACTION_SCHEMA, "action_shift": next(iter(shifts))}
        if self.normalization["split_hash"] != split["sha256"]:
            raise ValueError("归一化参数与数据划分不一致")
        if (self.normalization.get("action_schema") != ACTION_SCHEMA or
                self.normalization.get("action_shift") not in shifts):
            raise ValueError("checkpoint 与数据动作/输入历史 schema 不兼容")
        self.norm = normalization_arrays(self.normalization)
        self.optional_enabled = np.asarray(self.normalization["optional_state"]["counts"]) > 0
        LOGGER.info("Joint144 卡键投影：忽略 %d 行中的卡牌按钮；全部帧仍保留，源 NPZ 不变",
                    sum(row["card_input_ignored_rows"] for row in self.info.values()))

    def get(self, name):
        with self.lock:
            if name in self.cache:
                self.hits += 1
                self.cache.move_to_end(name)
                return self.cache[name]
            self.misses += 1
        shard = read_shard(self.root / name, self.config["data"]["vertical_positive_is_down"])
        size = sum(value.nbytes for value in shard.values())
        with self.lock:
            if name in self.cache:
                return self.cache[name]
            while self.cache and self.bytes + size > self.budget:
                _, old = self.cache.popitem(last=False)
                self.bytes -= sum(value.nbytes for value in old.values())
            # 单份超预算仍可处理，但不常驻缓存；预算是缓存上限，不是进程总内存上限。
            if size <= self.budget:
                self.cache[name] = shard
                self.bytes += size
        return shard

    def summary(self):
        result = {}
        for split in ("train", "validation"):
            rows = [self.info[name] for name in self.split[split]]
            result[split] = {"replays": len(rows), "frames": sum(x["frames"] for x in rows),
                             "transitions": sum(x["transitions"] for x in rows),
                             "joint_counts": np.sum([x["joint_counts"] for x in rows], 0).tolist(),
                             "optional_state_counts": np.sum([x["optional_state_counts"] for x in rows], 0).tolist()}
            result[split].update(self.action_history[split])
            result[split]["neutral_fraction"] = result[split]["joint_counts"][NEUTRAL_ACTION_ID] / result[split]["transitions"]
            result[split]["card_input_ignored_rows"] = sum(x["card_input_ignored_rows"] for x in rows)
            result[split]["resource_coverage"] = {
                side: np.sum([x["resource_coverage"][side] for x in rows], 0).tolist()
                for side in ("self", "opponent")
            }
        result["split_hash"] = self.split["sha256"]
        result["action_diagnostics_version"] = DIAGNOSTIC_VERSION
        result["raw_card_projection"] = RAW_CARD_PROJECTION
        return result

    def sample(self, rng, cfg, split="train"):
        names = self.split[split]
        weights = np.asarray([self.info[name]["transitions"] for name in names], np.float64)
        batch, length, burn = cfg["batch_size"], cfg["sequence_length"], cfg["burn_in"]
        # 每批从少量随机 REP 中取多段序列，减少压缩分片重复解压；跨批仍按转移数加权抽样。
        chosen = rng.choice(len(names), min(batch, cfg["replays_per_batch"]), p=weights / weights.sum())
        sizes = np.array_split(np.arange(batch), len(chosen))
        batches = []
        for shard_id, rows in zip(chosen, sizes):
            shard = self.get(names[shard_id])
            picks = rng.integers(0, shard["segment_cumulative"][-1], len(rows))
            segment = np.searchsorted(shard["segment_cumulative"], picks, side="right")
            starts, ends = shard["segments"][segment].T
            previous = np.r_[0, shard["segment_cumulative"][:-1]][segment]
            positions = starts + picks - previous
            burn_lengths = np.minimum(burn, positions - starts)
            prefix = positions[:, None] - burn_lengths[:, None] + np.arange(burn)
            # 纯 BC 只需 burn-in 和当前动作标签对应的状态，无未来前瞻状态。
            main = positions[:, None] + np.arange(length)
            indices = np.concatenate((np.minimum(prefix, positions[:, None]), np.minimum(main, ends[:, None] - 1)), 1)
            observation = self.observation(shard, indices)
            labels = np.minimum(main, ends[:, None] - 1)
            batches.append({"observation": observation, "burn_lengths": burn_lengths.astype(np.int64),
                            "joint_action_id": shard["joint_action_id"][labels],
                            "previous_expert_action_id": shard["previous_expert_action_id"][labels],
                            "mask": main < ends[:, None]})
        result = {key: np.concatenate([item[key] for item in batches]) for key in batches[0] if key != "observation"}
        result["observation"] = {key: np.concatenate([item["observation"][key] for item in batches])
                                 for key in batches[0]["observation"]}
        return result

    def observation(self, shard, indices):
        mean, std = self.norm["state"]
        categorical = shard["state_categorical"][indices].astype(np.int64)
        # 原始 NPZ 保持 0..21；只在模型入口压缩，避免改写既有数据集。
        categorical[..., -1] = compress_weather(categorical[..., -1])
        result = {"state_continuous": np.clip((shard["state_continuous"][indices] - mean) / std, -10, 10),
                  "state_categorical": categorical,
                  "tactical_state": shard["tactical_state"][indices].astype(np.float32)}
        result.update(resource_observation(shard, indices))
        result["previous_joint_action_id"] = shard["previous_joint_action_id"][indices]
        result["previous_action_duration"] = shard["previous_action_duration"][indices]
        mean, std = self.norm["optional_state"]
        mask = shard["state_optional_mask"][indices].astype(bool) & self.optional_enabled
        result["state_optional_mask"] = mask
        result["state_optional_continuous"] = np.where(mask,
            np.clip((shard["state_optional_continuous"][indices] - mean) / std, -10, 10), 0).astype(np.float32)
        mean, std = self.norm["objects"]
        for side in ("self", "opponent"):
            offsets = shard[f"{side}_object_offsets"]
            counts = offsets[indices + 1] - offsets[indices]
            mask = np.arange(3) < counts[..., None]
            selected = offsets[indices][..., None] + np.arange(3)
            numeric = np.zeros((*indices.shape, 3, 8), np.float32)
            category = np.zeros((*indices.shape, 3, 2), np.int64)
            numeric[mask] = np.clip((shard[f"{side}_object_numerical"][selected[mask]] - mean) / std, -10, 10)
            category[mask] = shard[f"{side}_object_categorical"][selected[mask]]
            result.update({f"{side}_object_numerical": numeric, f"{side}_object_categorical": category,
                           f"{side}_object_mask": mask})
        return result
