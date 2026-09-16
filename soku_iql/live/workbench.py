from __future__ import annotations

import copy
import queue
import threading
from concurrent.futures import TimeoutError

from ..config import resolve
from .agent import LiveAgent
from .config import validate
from .model_picker import ModelPicker
from .repository import Repository
from .runtime import LiveRuntime


class Workbench:
    """PPO 风格的工作台控制层；网页线程只提交命令，模型切换统一由协调线程串行执行。"""

    def __init__(self, config, auto_start=False):
        self.config = copy.deepcopy(config)
        self.repository = Repository(config)
        self.lock = threading.RLock()
        self.queue = queue.Queue()
        self.closed = threading.Event()
        self.runtime = None
        self.requests = {}
        self.busy = False
        self.operation = None
        self.operation_cancelled = threading.Event()
        self.picker = ModelPicker()
        self.selected_model = None
        self.error = None
        self.auto_start = auto_start
        self.thread = threading.Thread(target=self._run, name="bc-workbench-coordinator", daemon=False)

    def start(self):
        self.thread.start()

    def snapshot(self):
        with self.lock:
            runtime, busy, error = self.runtime, self.busy, self.error
            config = copy.deepcopy(self.config)
            operation, selected_model = self.operation, copy.deepcopy(self.selected_model)
        state = runtime.snapshot() if runtime else {"phase": "等待选择", "message": "点击顶部选择本机模型，再点击加载模型；无需指定命令行路径", "loaded": False}
        phase = state.get("phase")
        state.update(ui_mode="iql_live", workbench_busy=busy, workbench_error=error, runtime_config=config,
                     workbench_operation=operation, selected_model=selected_model,
                     state="error" if state.get("error") else "stopped" if phase == "已停止" else
                     "initializing" if phase in ("等待加载", "加载中") else
                     "evaluating" if state.get("control", {}).get("ready") else "paused")
        return state

    def submit(self, identifier, name, value):
        allowed = {"resume", "pause", "save", "stop", "load", "evaluate", "browse_model", "cancel_model_selection"}
        if name not in allowed:
            raise ValueError("实战只支持继续、暂停、保存报告、停止、加载模型和评估，不执行训练命令")
        with self.lock:
            if identifier in self.requests:
                return copy.deepcopy(self.requests[identifier])
            if self.closed.is_set():
                raise ValueError("工作台已关闭")
            if len(self.requests) >= 4096:
                raise ValueError("本会话命令记录已达上限，请重启工作台；不会删除去重记录后重复执行")
            if name == "cancel_model_selection":
                if self.operation == "browse_model":
                    self.operation_cancelled.set()
                    self.picker.cancel()
                result = {"id": identifier, "status": "done", "message": "已请求关闭文件选择窗口；原模型保持暂停"}
                self.requests[identifier] = result
                return copy.deepcopy(result)
            if self.busy and name != "stop":
                raise ValueError("正在处理控制请求，请等待完成；仍可停止或取消选文件")
            if name == "stop":
                self.auto_start = False
                self.operation_cancelled.set()
                self.picker.cancel()
            if name in ("load", "evaluate", "browse_model"):
                # 提交时即占用切换操作，防止双击或另一浏览器重复打开文件窗口。
                self.busy, self.operation = True, name
                self.auto_start = False
                self.operation_cancelled.clear()
                if name == "browse_model":
                    self.picker.prepare()
            if not self.busy:
                self.busy, self.operation = True, name
            result = {"id": identifier, "status": "queued", "message": "等待执行"}
            self.requests[identifier] = result
            self.queue.put((identifier, name, copy.deepcopy(value)))
            return copy.deepcopy(result)

    def request(self, identifier):
        with self.lock:
            return copy.deepcopy(self.requests.get(identifier))

    def _stop_runtime(self):
        runtime = self.runtime
        if runtime:
            runtime.command("stop")
            while runtime.thread and runtime.thread.is_alive():
                runtime.thread.join(timeout=0.1)

    def _new_runtime(self, config, prepared_agent=None):
        self._stop_runtime()
        if self.closed.is_set():
            return
        validate(config)
        runtime = LiveRuntime(config, prepared_agent=prepared_agent)
        with self.lock:
            self.config = copy.deepcopy(config)
            self.runtime = runtime
        runtime.start()

    def _control_command(self, name):
        if not self.runtime:
            raise ValueError("尚未创建实战会话")
        future = self.runtime.command(name)
        while True:
            try:
                return future.result(timeout=0.1)
            except TimeoutError:
                if self.closed.is_set() or not self.runtime.thread.is_alive():
                    raise ValueError("实战会话已停止，命令未执行")

    def _execute(self, name, value):
        if name == "browse_model":
            self._pause_for_model_selection()
            chosen = self.picker.choose(resolve(self.config["checkpoint"]))
            if chosen is None:
                return {"message": "已取消选择；没有更换模型，原会话保持暂停", "cancelled": True}
            model = self.repository.register_model(chosen)
            with self.lock:
                self.selected_model = copy.deepcopy(model)
            return {"message": "文件已选中，尚未切换；点击加载模型生效", "model": model}
        if name not in ("load", "evaluate"):
            if name == "stop":
                self._stop_runtime()
                return "已停止控制并保存评估报告；网页保留，可重新加载模型"
            return self._control_command(name)
        if not value.get("confirm_discard"):
            raise ValueError("请确认结束旧会话，未完小局将保存为片段")
        self._check_model_operation()
        if value.get("pause_before_load"):
            self._pause_for_model_selection()
        elif self.snapshot().get("control", {}).get("active"):
            raise ValueError("先暂停实战再切换模型/配置，不能在持键时替换模型")
        config = copy.deepcopy(self.config)
        model_id = value.get("model_id")
        if model_id:
            config["checkpoint"] = str(self.repository.path(model_id, "models"))
        for key in ("device", "rounds", "cpu_threads", "amp", "streaming_tcn", "tcn_cuda_graph"):
            if key in value:
                config[key] = value[key]
        if "player_side" in value:
            # 切侧复用完整会话重建流程，不能保留旧玩家的按键历史和 TCN 缓存。
            if value["player_side"] not in ("left", "right"):
                raise ValueError("观察侧必须是 left（1P）或 right（2P）")
            config["environment"]["player_side"] = value["player_side"]
        if "difficulty" in value:
            config["environment"]["cpu_difficulty_label"] = str(value["difficulty"])[:200]
        if "keyboard" in value:
            if not isinstance(value["keyboard"], dict) or set(value["keyboard"]) != set(config["keyboard"]):
                raise ValueError("必须提交完整的四方向和四按钮按键映射")
            config["keyboard"] = value["keyboard"]
        if "decision_interval_frames" in value:
            config["environment"]["decision_interval_frames"] = value["decision_interval_frames"]
        if "auto_restart" in value:
            config["restart"]["enabled"] = value["auto_restart"]
        validate(config)
        self._check_model_operation()
        # 先校验权重/输入规格并构建候选模型；失败时旧模型仍在，不丢失旧会话。
        try:
            candidate = LiveAgent(resolve(config["checkpoint"]), config)
        except Exception as exc:
            raise ValueError(f"新模型加载失败，未切换原模型：{exc}") from exc
        self._check_model_operation()
        self._new_runtime(config, prepared_agent=candidate)
        # 和 PPO 一样先完成载入，再允许开始；失败结果返回请求编号，不自动使用旧模型冒充。
        while not self.closed.wait(0.1):
            state = self.runtime.snapshot()
            if state.get("error") or not self.runtime.thread.is_alive():
                raise ValueError(state.get("message", "模型加载失败"))
            if state.get("loaded") and state.get("summary"):
                return "IQL/BC 策略已载入，保持暂停；确认游戏后点击继续开始评估"
        raise ValueError("工作台关闭，加载中止")

    def _check_model_operation(self):
        if self.closed.is_set() or self.operation_cancelled.is_set():
            raise ValueError("已取消模型操作，没有继续切换")

    def _pause_for_model_selection(self):
        self._check_model_operation()
        runtime = self.runtime
        if runtime and runtime.thread and runtime.thread.is_alive() and not runtime.closed.is_set():
            # 等待控制线程确认松键，不能只根据最多延迟 200ms 的网页快照判断。
            self._control_command("pause")
        self._check_model_operation()

    def _run(self):
        try:
            # 初始路径不存在时仍可进入工作台选文件，不让旧配置挡住图形化加载入口。
            if resolve(self.config["checkpoint"]).is_file():
                self._new_runtime(self.config)
            while not self.closed.is_set():
                if self.auto_start:
                    state = self.snapshot()
                    if state.get("loaded") and state.get("game", {}).get("pid"):
                        self.auto_start = False
                        self._control_command("resume")
                try:
                    identifier, name, value = self.queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                with self.lock:
                    self.busy, self.error = True, None
                    self.operation = name
                try:
                    message = self._execute(name, value)
                    result = {"id": identifier, "status": "done"}
                    result.update(message if isinstance(message, dict) else {"message": message})
                except Exception as exc:
                    result = {"id": identifier, "status": "failed", "message": str(exc)}
                with self.lock:
                    self.requests[identifier] = result
                    self.error = result["message"] if result["status"] == "failed" else None
                    self.busy = not self.queue.empty()
                    self.operation = "stop" if self.busy else None
        except Exception as exc:
            with self.lock:
                self.error = str(exc)
        finally:
            self.closed.set()
            self.picker.cancel()
            self._stop_runtime()
            with self.lock:
                for result in self.requests.values():
                    if result["status"] == "queued":
                        result.update(status="failed", message="工作台停止，命令未完成")

    def close(self):
        self.closed.set()
        self.picker.cancel()
        while self.thread.is_alive():
            self.thread.join(timeout=0.1)
