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
    parser.add_argument("--headless", action="store_true", help="无网页，立即开始离线训练")
    parser.add_argument("--start", action="store_true", help="网页模式准备完毕后自动开始")
    parser.add_argument("--port", type=int, default=8806, help="网页起始端口，冲突时自动递增")
    parser.add_argument("--host", choices=("127.0.0.1", "0.0.0.0"), default="127.0.0.1",
                        help="默认本机/SSH 转发；0.0.0.0 开放可信局域网，无 token")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not 1024 <= args.port <= 65535:
        parser.error("端口必须在 1024～65535")
    if args.headless:
        run(args.config, args.init_bc, args.resume)
    else:
        from soku_iql.workbench import Workbench
        from soku_iql.web_service import run_workbench
        run_workbench(Workbench(args.config, args.init_bc, args.resume, args.start), args.host, args.port)


if __name__ == "__main__":
    main()
