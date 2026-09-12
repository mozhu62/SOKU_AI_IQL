from __future__ import annotations

import argparse
import sys
import time

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from soku_iql.live.config import load_config, validate


def main():
    parser = argparse.ArgumentParser(description="IQL/BC 策略 Windows 实战验证；只推理，不训练")
    parser.add_argument("--config", default="configs/live_eval.yaml")
    parser.add_argument("--checkpoint", "--model", dest="checkpoint", help="可选初始 BC .pt；也可在网页顶部直接选择本机模型")
    parser.add_argument("--device", help="cpu / cuda / cuda:0 / auto")
    parser.add_argument("--rounds", type=int, help="完整小局数量，0 表示不限")
    parser.add_argument("--headless", action="store_true", help="无界面评估，连接游戏后自动控制；F10 暂停")
    parser.add_argument("--start", action="store_true", help="模型和游戏连接就绪后自动开始一次")
    parser.add_argument("--port", type=int, help="实战网页起始端口，默认 8826，冲突时自动递增")
    args = parser.parse_args()
    if sys.platform != "win32":
        parser.error("实战控制只能在 Windows 游戏电脑运行；Linux 可训练，将 .pt 复制回 Windows 后使用")
    config = load_config(args.config)
    for name in ("checkpoint", "device", "rounds"):
        if getattr(args, name) is not None:
            config[name] = getattr(args, name)
    if args.port is not None:
        config["web"]["port"] = args.port
    validate(config)
    if not args.headless:
        from soku_iql.live.web_service import WEB_DIST, run_workbench
        if not (WEB_DIST / "play.html").is_file():
            parser.error("实战网页尚未构建：请手动执行 npm --prefix web run build；默认已不再使用 Tk 窗口")
        run_workbench(config, args.start)
        return
    from soku_iql.live.runtime import LiveRuntime
    runtime = LiveRuntime(config)
    runtime.start()
    started = False
    try:
        while runtime.thread.is_alive():
            status = runtime.snapshot()
            if not started and status.get("loaded") and status.get("game", {}).get("pid"):
                runtime.command("resume")
                started = True
            print(status["phase"], status["message"], status.get("summary", {}), flush=True)
            summary = status.get("summary", {})
            if summary.get("target") and summary.get("completed", 0) >= summary["target"]:
                break
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        runtime.command("stop")
        # 停止前等松键与报告保存；不要用 terminate 杀掉正在持键的进程。
        while runtime.thread.is_alive():
            runtime.thread.join(timeout=0.2)
    final = runtime.snapshot()
    if final.get("error"):
        print(final["error"], file=sys.stderr)
        raise SystemExit(1)
    print("实战验证结束：", final.get("summary", {}).get("report_directory", ""))


if __name__ == "__main__":
    main()
