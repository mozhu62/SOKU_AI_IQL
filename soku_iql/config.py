from __future__ import annotations

import copy
import math
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
TRAINING = dict(device="auto", total_steps=100000, batch_size=16, sequence_length=32,
                burn_in=31, replays_per_batch=4, cpu_threads=4, amp=True,
                max_grad_norm=10.0, log_interval=20, save_interval=1000,
                validation_interval=1000, validation_batches=20, prefetch_batches=2)
IQL = dict(gamma=0.99, expectile=0.7, advantage_beta=3.0, max_weight=100.0,
           target_tau=0.005, actor_lr=0.0001, critic_lr=0.0003, value_lr=0.0003,
           weight_decay=0.0, actor_warmup_steps=1000)
REWARD = dict(damage_dealt=0.001, damage_taken=0.001)
ACTOR_SAMPLING = dict(enabled=False, neutral_max_fraction=0.25)


def resolve(path):
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def validate_resume(config, previous):
    for key in ("model", "seed", "iql", "reward"):
        if config[key] != previous[key]:
            raise ValueError(f"续训不能静默改变 {key}，请使用保存时的配置")
    # 验证采样依赖 batch_size/replays_per_batch；改变后不能再沿用旧 best_nll。
    for key in ("burn_in", "sequence_length", "batch_size", "replays_per_batch", "validation_batches"):
        if config["training"][key] != previous["training"][key]:
            raise ValueError(f"续训不能改变 training.{key}：需要保持训练/固定验证的采样口径")


def load(path, source):
    supplied = yaml.safe_load(resolve(path).read_text(encoding="utf-8-sig")) or {}
    allowed = {"data", "training", "iql", "reward", "output", "actor_sampling"}
    if not isinstance(supplied, dict) or set(supplied) - allowed:
        raise ValueError("IQL 配置只接受 data/training/iql/reward/output/actor_sampling；模型与 seed 继承 BC")
    cfg = dict(seed=source["seed"], model=copy.deepcopy(source["model"]),
               data=copy.deepcopy(source["data"]), training=copy.deepcopy(TRAINING),
               iql=copy.deepcopy(IQL), reward=copy.deepcopy(REWARD), actor_sampling=copy.deepcopy(ACTOR_SAMPLING),
               output=dict(directory="outputs/iql_suika"))
    for section, values in supplied.items():
        if not isinstance(values, dict) or set(values) - set(cfg[section]):
            raise ValueError(f"未知 {section} 参数")
        cfg[section].update(values)
    for key in ("directory", "split_file"):
        cfg["data"][key] = str(resolve(cfg["data"][key]))
    cfg["output"]["directory"] = str(resolve(cfg["output"]["directory"]))
    if cfg["data"]["train_fraction"] != .8 or cfg["data"]["vertical_positive_is_down"] != source["data"]["vertical_positive_is_down"]:
        raise ValueError("BC→IQL 不允许改变划分比例或方向编码")
    t, q = cfg["training"], cfg["iql"]
    sampling = cfg["actor_sampling"]
    fraction = sampling["neutral_max_fraction"]
    if type(sampling["enabled"]) is not bool or type(fraction) not in (int, float) or not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("actor_sampling.enabled 必须为布尔值，neutral_max_fraction 必须为 [0,1] 有限数值")
    if not isinstance(t["device"], str) or (t["device"] not in ("auto", "cpu", "cuda")
            and not (t["device"].startswith("cuda:") and t["device"][5:].isdigit())):
        raise ValueError("training.device 只允许 auto、cpu、cuda 或 cuda:N")
    for key in ("total_steps", "batch_size", "sequence_length", "replays_per_batch", "cpu_threads",
                "log_interval", "save_interval", "validation_interval", "validation_batches", "prefetch_batches"):
        if type(t[key]) is not int or t[key] < 1:
            raise ValueError(f"training.{key} 必须为正整数")
    if t["burn_in"] != 31 or t["prefetch_batches"] > 8 or type(t["amp"]) is not bool:
        raise ValueError("TCN32 必须为 burn_in=31，预取不超过 8，amp 必须是布尔值")
    for key, value in q.items():
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"iql.{key} 必须为有限数值")
    if not (0 < q["gamma"] <= 1 and .5 < q["expectile"] < 1 and 0 < q["target_tau"] <= 1
            and q["advantage_beta"] >= 0 and q["max_weight"] >= 1 and q["weight_decay"] >= 0
            and all(0 < q[k] <= .01 for k in ("actor_lr", "critic_lr", "value_lr"))):
        raise ValueError("IQL 折扣、expectile、EMA、优势权重或学习率超范围")
    if type(q["actor_warmup_steps"]) is not int or q["actor_warmup_steps"] < 0:
        raise ValueError("actor_warmup_steps 必须为非负整数")
    for value in (*cfg["reward"].values(), t["max_grad_norm"], cfg["data"]["cache_gb"]):
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("奖励系数、梯度上限和缓存必须为非负有限数值")
    if t["max_grad_norm"] <= 0 or not .1 <= cfg["data"]["cache_gb"] <= 1024:
        raise ValueError("梯度上限或缓存容量无效")
    return cfg
