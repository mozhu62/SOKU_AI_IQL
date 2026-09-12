from __future__ import annotations

import os
import tempfile
from pathlib import Path

import torch

from .config import NETWORK_VERSION, MODEL_DEFAULTS, network_version_for, palr_settings
from .models import network_spec
from .action_space import ACTION_SCHEMA
from .schema import policy_input_manifest


def load(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"未找到 BC checkpoint：{path}；从零训练请去掉 --resume")
    package = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(package, dict) or package.get("algorithm") != "bc":
        raise ValueError("仅接受 BC checkpoint；CQL/PPO/IQL 权重不能作为 BC 续训模型，请从随机初始化开始")
    version = package.get("network_version")
    if version not in (NETWORK_VERSION, "soku_iql_tcn64_joint144_v1", "soku_iql_tcn256_joint144_v1"):
        raise ValueError(
            "BC checkpoint schema 不兼容：当前版本为 228D 状态、256D 当前编码和 Joint144 输出，"
            "已移除卡牌输入/输出和技能等级；旧 Joint432 权重不允许部分加载。"
            "请去掉 --resume，从随机初始化开始并使用新输出目录"
        )
    if package.get("spec", {}).get("network_version") != version:
        raise ValueError("checkpoint 顶层网络版本与 spec 不一致")
    if (package["spec"].get("action_schema") != ACTION_SCHEMA or
            package["spec"].get("inputs") != policy_input_manifest()):
        raise ValueError("checkpoint action/observation schema 不兼容，不允许部分加载")
    if package["spec"].get("output_semantics") != "categorical_logits":
        raise ValueError("checkpoint 输出不是 BC 分类 logits")
    required = {"model", "optimizer", "scaler", "config", "split_hash", "normalization", "step",
                "updates", "samples", "best", "stage", "rng_cpu", "rng_cuda"}
    if required - package.keys():
        raise ValueError(f"BC checkpoint 缺少字段：{sorted(required - package.keys())}")
    model = {**MODEL_DEFAULTS, **package["spec"]["model"]}
    if version != network_version_for(model):
        raise ValueError("checkpoint 的时序结构与网络版本不一致")
    canonical = network_spec(model)
    if "temporal" in package["spec"] and package["spec"]["temporal"] != canonical["temporal"]:
        raise ValueError("checkpoint 时序窗口或记忆语义不兼容")
    if "temporal" not in package["spec"]:
        raise ValueError("checkpoint 缺少明确的 32 帧时序协议")
    config_model = {**MODEL_DEFAULTS, **package["config"]["model"]}
    if config_model != model:
        raise ValueError("checkpoint 配置与模型结构清单不一致")
    package["config"]["model"] = config_model
    # PALR 没有权重；旧同架构 checkpoint 缺少此段时明确恢复为关闭状态。
    package["config"]["palr"] = palr_settings(package["config"].get("palr"))
    package["spec"] = canonical
    return package


def save(path, learner, config, split, normalization, step, updates, samples, best, stage):
    def cpu(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {key: cpu(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cpu(item) for item in value]
        if isinstance(value, tuple):
            return tuple(cpu(item) for item in value)
        return value
    package = {"algorithm": "bc", "network_version": learner.model.spec["network_version"], "spec": learner.model.spec,
               "model": cpu(learner.model.state_dict()),
               "optimizer": cpu(learner.optimizer.state_dict()), "scaler": learner.scaler.state_dict(),
               "config": config,
               "split_hash": split["sha256"], "normalization": normalization,
               "step": step, "updates": updates, "samples": samples, "best": best, "stage": stage,
               "rng_cpu": torch.get_rng_state(),
               "rng_cuda": torch.cuda.get_rng_state_all() if learner.device.type == "cuda" else []}
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(handle)
    try:
        torch.save(package, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def restore(package, learner, split):
    if package["spec"] != learner.model.spec or package["split_hash"] != split["sha256"]:
        raise ValueError("checkpoint 的网络/动作/输入结构或固定数据划分不兼容")
    learner.model.load_state_dict(package["model"], strict=True)
    learner.optimizer.load_state_dict(package["optimizer"])
    learner.scaler.load_state_dict(package["scaler"])
    learner.apply_settings(learner.config)
    torch.set_rng_state(package["rng_cpu"].cpu().to(torch.uint8))
    if learner.device.type == "cuda" and len(package["rng_cuda"]) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all([value.cpu().to(torch.uint8) for value in package["rng_cuda"]])
