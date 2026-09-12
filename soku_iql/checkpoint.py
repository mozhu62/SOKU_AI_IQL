from __future__ import annotations

import copy
import hashlib
import os
import tempfile
from pathlib import Path

import torch

from .bc_core.models import network_spec

VERSION = "soku_iql_bc_tcn32_joint144_v1"
VERSION64 = "soku_iql_bc_tcn64_joint144_v1"
VERSION256 = "soku_iql_bc_tcn256_joint144_v1"


def version_for(model):
    return {"tcn": VERSION, "tcn64": VERSION64, "tcn256": VERSION256}[model["temporal_mode"]]


def load_source(path, *, bc=False):
    """把来源哈希与实际加载的版本绑定，拒绝采集数据期间被覆盖的 last.pt。"""
    from .bc_core import checkpoint as bc_checkpoint
    path = Path(path).resolve()
    def signature():
        stat = path.stat()
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns
    before = signature()
    package = bc_checkpoint.load(path) if bc else load(path)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if signature() != before:
        raise ValueError("加载期间 checkpoint 被更新，请先复制为固定文件后再迁移/续训")
    return package, digest


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


def atomic_save(path, package):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    os.close(handle)
    try:
        torch.save(package, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load(path):
    package = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(package, dict) or package.get("algorithm") != "iql" or package.get("version") not in (VERSION, VERSION64, VERSION256):
        raise ValueError("仅接受当前 IQL 续训包；BC 请使用 --init-bc")
    required = {"networks", "optimizers", "scalers", "config", "bc_config", "normalization", "split_hash",
                "step", "samples", "actor_updates", "best_nll", "rng_cpu", "rng_cuda", "spec", "provenance"}
    if required - package.keys() or package["spec"] != network_spec(package["config"]["model"]):
        raise ValueError("IQL checkpoint 不完整或 BC 架构不匹配")
    if package["version"] != version_for(package["config"]["model"]):
        raise ValueError("IQL 包版本与时序结构不一致")
    return package


def save(path, learner, bc_config, normalization, split_hash, step, samples, actor_updates, best_nll, provenance):
    atomic_save(path, dict(algorithm="iql", version=version_for(learner.config["model"]), spec=learner.networks.actor.spec,
                          networks=cpu(learner.networks.state_dict()),
                          optimizers={k: cpu(v.state_dict()) for k, v in learner.optimizers.items()},
                          scalers={k: v.state_dict() for k, v in learner.scalers.items()},
                          config=learner.config, bc_config=bc_config, normalization=normalization,
                          split_hash=split_hash, step=step, samples=samples, actor_updates=actor_updates,
                          best_nll=best_nll, provenance=provenance, rng_cpu=torch.get_rng_state(),
                          rng_cuda=torch.cuda.get_rng_state_all() if learner.device.type == "cuda" else []))


def export_actor(path, learner, bc_config, normalization, split_hash, step, samples, actor_updates, provenance):
    # 原 BC 加载器要求完整字段；只放策略优化器，绝不把 Q/V 或 target 塞入推理包。
    config = copy.deepcopy(bc_config)
    config["model"] = copy.deepcopy(learner.config["model"])
    config["data"] = copy.deepcopy(learner.config["data"])
    config["training"].update(learning_rate=learner.config["iql"]["actor_lr"],
                              burn_in=learner.config["training"]["burn_in"],
                              weight_decay=learner.config["iql"]["weight_decay"], frozen_modules=[])
    config.setdefault("palr", {})["enabled"] = False
    config.setdefault("keyframe_weighting", {})["enabled"] = False
    spec = learner.networks.actor.spec
    atomic_save(path, dict(algorithm="bc", network_version=spec["network_version"], spec=spec,
                          model=cpu(learner.networks.actor.state_dict()),
                          optimizer=cpu(learner.optimizers["actor"].state_dict()),
                          scaler=learner.scalers["actor"].state_dict(), config=config, split_hash=split_hash,
                          normalization=normalization, step=step, updates=actor_updates, samples=samples,
                          best=None, stage=0, rng_cpu=torch.get_rng_state(),
                          rng_cuda=torch.cuda.get_rng_state_all() if learner.device.type == "cuda" else [],
                          provenance={**provenance, "trained_algorithm": "iql", "export_format": "bc_policy_compatible"}))
