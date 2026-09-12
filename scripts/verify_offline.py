"""真实 BC 模型的短步离线验收；只写独立验收目录，不启动游戏。"""
import argparse
import copy
import gc
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from soku_iql import checkpoint
from soku_iql.bc_core.storage import atomic_json
from soku_iql.config import load
from soku_iql.dataset import IQLReplayStore
from soku_iql.learner import tensor_batch
from soku_iql.runtime import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-bc", required=True)
    parser.add_argument("--config", default="configs/iql_suika.yaml")
    parser.add_argument("--bc-project", default=str(ROOT.parent / "soku_bc"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    output = ROOT / "outputs" / ("offline_acceptance_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(output / "acceptance.log", encoding="utf-8")])
    source, source_hash = checkpoint.load_source(args.init_bc, bc=True)
    cfg = load(args.config, source["config"])
    short = {key: copy.deepcopy(cfg[key]) for key in ("data", "training", "iql", "reward", "output")}
    short["data"]["cache_gb"] = .5
    short["training"].update(device=args.device, batch_size=2, sequence_length=4, replays_per_batch=2,
                             total_steps=1, log_interval=1, save_interval=1,
                             validation_interval=1, validation_batches=1)
    short["iql"]["actor_warmup_steps"] = 1
    short["output"]["directory"] = str(output / "run")
    config_path = output / "acceptance_config.json"
    atomic_json(config_path, short)
    # 首步只更新 Q/V，第二次调用正式入口恢复完整状态，再更新两步 Actor。
    run(str(config_path), init_bc=args.init_bc)
    saved = checkpoint.load(output / "run" / "last.pt")
    assert saved["step"] == 1 and saved["actor_updates"] == 0
    assert all(torch.equal(v, saved["networks"]["actor." + k]) for k, v in source["model"].items())
    assert saved["optimizers"]["actor"]["state"] == {}
    assert saved["optimizers"]["critic"]["state"] and saved["optimizers"]["value"]["state"]
    del saved
    gc.collect()
    short["training"]["total_steps"] = 3
    atomic_json(config_path, short)
    run(str(config_path), resume=str(output / "run" / "last.pt"))
    saved = checkpoint.load(output / "run" / "last.pt")
    assert saved["step"] == 3 and saved["actor_updates"] == 2
    assert any(not torch.equal(v, saved["networks"]["actor." + k]) for k, v in source["model"].items())
    optimizer_steps = {name: sorted({int(state["step"]) for state in opt["state"].values()})
                       for name, opt in saved["optimizers"].items()}
    assert optimizer_steps == {"actor": [2], "critic": [3], "value": [3]}, optimizer_steps
    # 使用原项目加载器与原网络验收，不以私有副本的自洽性替代真实兼容性。
    sys.path.insert(0, str(Path(args.bc_project).resolve()))
    from soku_bc.checkpoint import load as original_load
    from soku_bc.models import BCNetwork as OriginalBCNetwork
    from soku_iql.bc_core.models import BCNetwork
    exported = original_load(output / "run" / "actor_bc.pt")
    original = OriginalBCNetwork(exported["config"]["model"]).eval()
    original.load_state_dict(exported["model"], strict=True)
    policy = BCNetwork(saved["config"]["model"]).eval()
    policy.load_state_dict({k.removeprefix("actor."): v for k, v in saved["networks"].items()
                            if k.startswith("actor.")}, strict=True)
    initial = original_load(output / "run" / "bc_initial_actor.pt")
    assert all(torch.equal(v, initial["model"][k]) for k, v in source["model"].items())
    split = json.loads(Path(cfg["data"]["split_file"]).read_text(encoding="utf-8"))
    # 正式 run 已校验全部分片；这里只抽两份真实分片做导出前后同输入比对。
    subset = copy.deepcopy(split)
    subset["train"], subset["validation"] = split["train"][:1], split["validation"][:1]
    subset["files"] = {k: split["files"][k] for k in subset["train"] + subset["validation"]}
    store = IQLReplayStore(saved["config"], subset, logging.info, normalization=source["normalization"])
    batch = tensor_batch(store.sample(np.random.default_rng(123), short["training"], "validation"), "cpu")
    with torch.no_grad():
        expected = policy(batch["observation"], 31, batch["burn_lengths"])
        actual = original(batch["observation"], 31, batch["burn_lengths"])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    _, final_hash = checkpoint.load_source(args.init_bc, bc=True)
    assert final_hash == source_hash
    assert exported["normalization"] == source["normalization"]
    result = dict(passed=True, source=str(Path(args.init_bc).resolve()), source_sha256=source_hash,
                  source_step=source["step"], split_hash=split["sha256"],
                  train_shards=len(split["train"]), validation_shards=len(split["validation"]),
                  device=args.device, torch_version=str(torch.__version__),
                  gpu=torch.cuda.get_device_name(0) if args.device.startswith("cuda") else None,
                  completed_steps=saved["step"], actor_updates=saved["actor_updates"],
                  optimizer_steps=optimizer_steps, warmup_actor_unchanged=True,
                  original_bc_loader_compatible=True, export_logits_max_abs_error=float((actual-expected).abs().max()),
                  source_unchanged=True, normalization_unchanged=True,
                  validation_scope="每次固定一批、至多8个有效转移；不代表全量验证或实战提升",
                  output=str(output))
    atomic_json(output / "result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
