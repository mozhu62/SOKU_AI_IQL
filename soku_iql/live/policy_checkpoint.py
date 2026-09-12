"""实战仅提取 Actor；兼容 IQL 完整包、导出策略和原 BC 基线。"""
from pathlib import Path

import torch

from .. import checkpoint
from ..bc_core import checkpoint as bc_checkpoint


def load(path: Path):
    package = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(package, dict):
        raise ValueError("不是模型 checkpoint")
    if package.get("algorithm") == "bc":
        return bc_checkpoint.load(path)
    del package
    source = checkpoint.load(path)
    spec = source["spec"]
    # 不构造 Q/V，不恢复优化器，不创建转换副本；推理只占用原 BC 策略大小的网络。
    actor = {key.removeprefix("actor."): value for key, value in source["networks"].items() if key.startswith("actor.")}
    return dict(model=actor, spec=spec, network_version=spec["network_version"], config=source["config"],
                normalization=source["normalization"], step=source["step"],
                provenance={**source["provenance"], "trained_algorithm": "iql"})
