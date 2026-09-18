from __future__ import annotations

import copy
import math
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
NETWORK_VERSION = "soku_bc_tcn32_joint144_weather9_v2"
MODEL_DEFAULTS = {
    "action_vocab_size": 2048, "block_vocab_size": 512, "weather_vocab_size": 9,
    "action_embedding_dim": 32, "block_embedding_dim": 8, "weather_embedding_dim": 8,
    "current_hidden_dim": 256, "object_hidden_dim": 64, "object_set_dim": 128,
    "temporal_hidden_dim": 256, "temporal_output_dim": 256, "fusion_dim": 1024,
    "object_embedding_mode": "separate",
    "skill_embedding_dim": 8, "previous_action_embedding_dim": 32,
    "temporal_mode": "tcn",
}
KEYFRAME_DEFAULTS = {"enabled": False, "changepoint_weight": 4.0}
PALR_DEFAULTS = {"enabled": False, "alpha": 0.1, "sample_size": 256,
                 "feature_kernel": "rbf", "action_kernel": "categorical", "regularization": 0.001}
DEFAULTS = {
    "seed": 42,
    "data": {"directory": "../soku_cql/data/replay_shards_resources_v4",
             "split_file": "data/train_val_split_resources_v4.json",
             "train_fraction": 0.8, "cache_gb": 4.0, "vertical_positive_is_down": True},
    "model": MODEL_DEFAULTS,
    "keyframe_weighting": KEYFRAME_DEFAULTS,
    "palr": PALR_DEFAULTS,
    "training": {"device": "auto", "total_steps": 100000, "batch_size": 32,
                 "sequence_length": 32, "burn_in": 31, "replays_per_batch": 4,
                 "learning_rate": 0.0001, "label_smoothing": 0.0, "max_grad_norm": 10.0,
                 "weight_decay": 0.0001, "log_interval": 20, "save_interval": 1000,
                 "validation_interval": 1000, "validation_batches": 20,
                 "cpu_threads": 4, "amp": True, "prefetch_batches": 2, "frozen_modules": []},
    "output": {"directory": "outputs/bc_suika_tcn32_joint144"},
    "web": {"host": "127.0.0.1", "port": 8796, "port_attempts": 30},
}
MODULES = ("current_encoder", "object_encoder", "tcn", "fusion", "policy_head")


def network_version_for(model):
    if model.get("temporal_mode", "tcn") == "tcn256":
        return "soku_iql_tcn256_joint144_weather9_v2"
    if model.get("temporal_mode", "tcn") == "tcn64":
        return "soku_iql_tcn64_joint144_weather9_v2"
    if model.get("temporal_mode", "tcn") != "tcn":
        raise ValueError("当前 BC 网络已删除 GRU，model.temporal_mode 必须为 tcn")
    return NETWORK_VERSION


def context_frames_for(model):
    network_version_for(model)
    return {"tcn": 32, "tcn64": 64, "tcn256": 256}[model.get("temporal_mode", "tcn")]


def active_modules(model):
    network_version_for(model)
    return MODULES
# 运行中可调项只在暂停后应用；改变输入、结构和数据划分必须另开训练。
EDITABLE = {
    "learning_rate": (1e-8, 0.01, "float", "学习率"),
    "label_smoothing": (0.0, 0.2, "float", "标签平滑（0 为标准 BC）"),
    "max_grad_norm": (0.01, 1000.0, "float", "梯度裁剪上限"),
    "weight_decay": (0.0, 0.1, "float", "权重衰减"),
    "batch_size": (1, 1024, "int", "每批序列数"),
    "total_steps": (1, 100000000, "int", "总更新步数"),
    "log_interval": (1, 10000, "int", "指标记录间隔"),
    "save_interval": (1, 1000000, "int", "模型保存间隔"),
    "validation_interval": (1, 1000000, "int", "验证间隔"),
    "validation_batches": (1, 1000, "int", "固定验证批次数"),
}


def resolve(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (ROOT / candidate).resolve()


def keyframe_weighting_settings(settings=None):
    # 旧配置缺少此段时保持等权 BC；显式配置后再启用关键帧加权。
    if settings is None:
        settings = {}
    if not isinstance(settings, dict) or set(settings) - set(KEYFRAME_DEFAULTS):
        raise ValueError("keyframe_weighting 只接受 enabled 和 changepoint_weight")
    result = {**KEYFRAME_DEFAULTS, **settings}
    if type(result["enabled"]) is not bool:
        raise ValueError("keyframe_weighting.enabled 必须为布尔值")
    weight = result["changepoint_weight"]
    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight < 1:
        raise ValueError("keyframe_weighting.changepoint_weight 必须为大于等于 1 的有限数值")
    return result


def palr_settings(settings=None):
    if settings is None:
        settings = {}
    if not isinstance(settings, dict) or set(settings) - set(PALR_DEFAULTS):
        raise ValueError("palr 包含未知配置项")
    result = {**PALR_DEFAULTS, **settings}
    if type(result["enabled"]) is not bool:
        raise ValueError("palr.enabled 必须为布尔值")
    for key, minimum in (("alpha", 0.0), ("regularization", 1e-6)):
        value = result[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < minimum:
            raise ValueError(f"palr.{key} 必须为不小于 {minimum} 的有限数值")
    if type(result["sample_size"]) is not int or not 2 <= result["sample_size"] <= 4096:
        raise ValueError("palr.sample_size 必须为 2～4096 的整数；建议先使用 256")
    if result["feature_kernel"] != "rbf" or result["action_kernel"] != "categorical":
        raise ValueError("PALR 本版只支持 RBF 特征核与 categorical 动作核")
    return result


def validate(config: dict) -> dict:
    config["keyframe_weighting"] = keyframe_weighting_settings(config.get("keyframe_weighting"))
    config["palr"] = palr_settings(config.get("palr"))
    # 结构字段统一补齐后再校验；旧 GRU 配置会因模式或宽度不兼容而被明确拒绝。
    unknown_model = set(config["model"]) - set(MODEL_DEFAULTS)
    if unknown_model:
        raise ValueError(f"model 包含旧架构或未知字段：{sorted(unknown_model)}")
    config["model"] = {**MODEL_DEFAULTS, **config["model"]}
    network_version_for(config["model"])
    if isinstance(config["seed"], bool) or not isinstance(config["seed"], int) or config["seed"] < 0:
        raise ValueError("seed 必须为非负整数")
    cfg = config["training"]
    for key, (lo, hi, kind, _) in EDITABLE.items():
        value = cfg[key]
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not lo <= value <= hi
                or (kind == "int" and not isinstance(value, int))):
            raise ValueError(f"training.{key} 必须为 {lo} 到 {hi} 之间的 {kind}")
    for key, lo in (("sequence_length", 1), ("burn_in", 0), ("replays_per_batch", 1), ("cpu_threads", 1)):
        if type(cfg[key]) is not int or not lo <= cfg[key] <= 4096:
            raise ValueError(f"training.{key} 超出范围")
    if cfg["burn_in"] != context_frames_for(config["model"]) - 1:
        raise ValueError("training.burn_in 必须等于 TCN 上下文长度减一，不是 GRU 预热")
    if type(cfg["amp"]) is not bool or type(cfg["prefetch_batches"]) is not int or not 1 <= cfg["prefetch_batches"] <= 8:
        raise ValueError("amp 必须为布尔值，prefetch_batches 必须在 1 到 8 之间")
    frozen = cfg["frozen_modules"]
    if (not isinstance(frozen, list) or any(x not in MODULES for x in frozen)
            or set(active_modules(config["model"])) <= set(frozen)):
        raise ValueError("冻结模块无效或全部模块均被冻结")
    if config["data"]["train_fraction"] != 0.8:
        raise ValueError("本版本固定按整份 REP 进行 8:2 划分")
    cache_gb = config["data"]["cache_gb"]
    if type(cache_gb) not in (int, float) or not math.isfinite(cache_gb) or not 0.1 <= cache_gb <= 1024:
        raise ValueError("data.cache_gb 必须在 0.1 到 1024 之间")
    if type(config["data"]["vertical_positive_is_down"]) is not bool:
        raise ValueError("vertical_positive_is_down 必须是布尔值")
    model = config["model"]
    for key, default in MODEL_DEFAULTS.items():
        if isinstance(default, int) and (type(model[key]) is not int or not 1 <= model[key] <= 65536):
            raise ValueError(f"model.{key} 必须为正整数")
    if model["object_set_dim"] != model["object_hidden_dim"] * 2:
        raise ValueError("object_set_dim 必须等于 object_hidden_dim 的两倍")
    if model["object_embedding_mode"] not in ("shared", "separate"):
        raise ValueError("object_embedding_mode 无效")
    if model["weather_vocab_size"] != MODEL_DEFAULTS["weather_vocab_size"]:
        raise ValueError("当前架构固定将天气压缩为 normal+八种特殊天气，weather_vocab_size 必须为 9")
    for key in ("current_hidden_dim", "object_hidden_dim", "object_set_dim", "temporal_hidden_dim",
                "temporal_output_dim", "fusion_dim"):
        if model[key] != MODEL_DEFAULTS[key]:
            raise ValueError(f"当前 TCN32 架构固定 model.{key}={MODEL_DEFAULTS[key]}")
    if config["web"]["host"] not in ("0.0.0.0", "127.0.0.1"):
        raise ValueError("web.host 只允许 0.0.0.0 或 127.0.0.1")
    if (type(config["web"]["port"]) is not int or type(config["web"]["port_attempts"]) is not int
            or not 1024 <= config["web"]["port"] <= 65535 or not 1 <= config["web"]["port_attempts"] <= 100):
        raise ValueError("本机端口或尝试数量无效")
    return config


def load_config(path: str | Path) -> dict:
    supplied = yaml.safe_load(resolve(path).read_text(encoding="utf-8-sig")) or {}
    result = copy.deepcopy(DEFAULTS)
    for key, value in supplied.items():
        if key not in result:
            raise ValueError(f"未知配置项：{key}")
        if isinstance(result[key], dict):
            if not isinstance(value, dict) or set(value) - set(result[key]):
                raise ValueError(f"{key} 包含未知配置项")
            result[key].update(value)
        else:
            result[key] = value
    return validate(result)
