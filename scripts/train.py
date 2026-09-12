import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from soku_iql.runtime import run


def main():
    parser = argparse.ArgumentParser(description="BC→IQL 离线训练；不启动游戏")
    parser.add_argument("--config", default="configs/iql_suika.yaml")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--init-bc")
    group.add_argument("--resume")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(args.config, args.init_bc, args.resume)


if __name__ == "__main__":
    main()
