import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from soku_iql import checkpoint
from soku_iql.learner import Learner


def main():
    parser = argparse.ArgumentParser(description="IQL 完整续训包导出为原 BC 推理端可读策略")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source, output = Path(args.checkpoint).resolve(), Path(args.output).resolve()
    if source == output or output.exists():
        raise ValueError("导出必须使用不存在的新文件，不能覆盖源模型")
    package, _ = checkpoint.load_source(source)
    config = package["config"]
    config["training"]["device"] = "cpu"
    learner = Learner(config)
    learner.networks.load_state_dict(package["networks"], strict=True)
    for key in learner.optimizers:
        learner.optimizers[key].load_state_dict(package["optimizers"][key])
        learner.scalers[key].load_state_dict(package["scalers"][key])
    checkpoint.export_actor(output, learner, package["bc_config"], package["normalization"], package["split_hash"],
                            package["step"], package["samples"], package["actor_updates"], package["provenance"])
    print(f"已导出 BC 推理兼容策略：{output}")


if __name__ == "__main__":
    main()
