"""离线训练工作台：网页只提交命令，所有训练修改在更新边界执行。"""
from __future__ import annotations

import copy
import json
import logging
import math
import queue
import threading
import time
from collections import deque
from pathlib import Path

from .bc_core.storage import atomic_json
from .config import resolve

# 不开放输入结构、采样口径、奖励和折扣的在线修改，避免旧 best/回报定义混用。
EDITABLE = {
    "training.total_steps": (1, 100000000, "int", "总更新步数"),
    "training.log_interval": (1, 10000, "int", "日志间隔"),
    "training.save_interval": (1, 100000, "int", "保存间隔"),
    "training.validation_interval": (1, 100000, "int", "验证间隔"),
    "training.prefetch_batches": (1, 8, "int", "预取批次数"),
    "training.cpu_threads": (1, 64, "int", "PyTorch CPU 线程"),
    "data.cache_gb": (.1, 1024, "float", "分片内存缓存 GiB"),
}


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    return value


class Workbench:
    def __init__(self, config_path, init_bc=None, resume=None, auto_start=False):
        self.config_path, self.init_bc, self.resume_path = config_path, init_bc, resume
        self.lock = threading.RLock()
        self.queue = queue.Queue()
        self.stop_event = threading.Event()
        self.running = auto_start
        self.requests = {}
        self.rows = {key: deque(maxlen=2000) for key in ("train", "validation", "configuration")}
        self.totals = {key: 0 for key in self.rows}
        self.recent = deque(maxlen=20)
        self.stage = 0
        self._progress_at = 0.
        self.status = dict(state="initializing", message="等待准备数据", config={}, config_source=str(resolve(config_path)),
                           output=None, step=0, samples=0, actor_updates=0, ready=False, error=None,
                           stage=0, timings={}, latest_train=None, latest_validation=None, speed={})
        self.thread = threading.Thread(target=self._run, name="iql-training", daemon=False)

    def start(self):
        self.thread.start()

    def snapshot(self):
        with self.lock:
            return copy.deepcopy(self.status)

    def publish(self, **values):
        with self.lock:
            self.status.update(copy.deepcopy(json_safe(values)))

    def attach(self, config, source):
        self.output = Path(config["output"]["directory"])
        self.publish(config=config, output=str(self.output), source_model=str(source))
        # 历史只在准备时读取一次；浏览器刷新只获取有上限的内存摘要。
        for kind in self.rows:
            path = self.output / f"{kind}.jsonl"
            if path.is_file():
                with path.open(encoding="utf-8") as stream:
                    for line in stream:
                        try:
                            row = json.loads(line)
                        except ValueError:
                            continue
                        self.record(kind, row)
                        self.stage = max(self.stage, int(row.get("stage", 0)))
                if kind != "configuration" and self.rows[kind]:
                    self.publish(**{f"latest_{kind}": self.rows[kind][-1]})
        self.publish(stage=self.stage)

    def progress(self, message):
        now = time.monotonic()
        if now - self._progress_at > .2:
            self._progress_at = now
            self.publish(message=message)

    def timing(self, key, seconds):
        with self.lock:
            times = self.status["timings"]
            times[key] = times.get(key, 0.) + seconds

    def record(self, kind, row):
        with self.lock:
            row = json_safe(row)
            self.rows[kind].append(copy.deepcopy(row))
            self.totals[kind] += 1
            if kind != "configuration":
                self.status[f"latest_{kind}"] = copy.deepcopy(row)

    def history_page(self, kind, offset=0, limit=300):
        if kind not in self.rows:
            raise ValueError("未知历史类型")
        with self.lock:
            rows = list(self.rows[kind])
            end = max(0, len(rows) - offset)
            return dict(rows=rows[max(0, end-limit):end], total=self.totals[kind], retained=len(rows),
                        truncated=self.totals[kind] > len(rows))

    def update(self, step, samples, actor_updates, result, wait):
        self.recent.append((result["samples"], wait + result["optimization_seconds"]))
        elapsed = sum(x[1] for x in self.recent)
        self.timing("data_wait", wait)
        self.timing("optimization", result["optimization_seconds"])
        self.publish(step=step, samples=samples, actor_updates=actor_updates,
                     latest_train={**result, "step": step, "stage": self.stage},
                     speed=dict(steps_per_second=len(self.recent)/max(elapsed, 1e-9),
                                samples_per_second=sum(x[0] for x in self.recent)/max(elapsed, 1e-9),
                                window_steps=len(self.recent)))

    def submit(self, identifier, name, value):
        if name not in ("resume", "pause", "save", "validate", "apply", "stop"):
            raise ValueError("未知训练命令")
        if not isinstance(value, dict):
            raise ValueError("命令参数必须是对象")
        with self.lock:
            if identifier in self.requests:
                return copy.deepcopy(self.requests[identifier])
            if self.status["state"] in ("error", "stopped", "completed") or self.stop_event.is_set():
                raise ValueError("训练线程已结束或正在停止，请重新启动服务续训")
            if len(self.requests) >= 4096:
                raise ValueError("本次会话命令数量达到上限，请保存后重启")
            if name != "stop" and not self.status["ready"]:
                raise ValueError("数据和模型尚未准备完毕")
            if name in ("apply", "validate") and self.status["state"] != "paused":
                raise ValueError("请先暂停，等待当前更新结束")
            result = dict(id=identifier, status="queued", message="等待训练边界执行", command=name)
            self.requests[identifier] = result
            if name == "stop":
                self.stop_event.set()
                self.status["message"] = "正在停止；若模型已就绪，将在完整更新结束后保存"
            self.queue.put((identifier, name, copy.deepcopy(value)))
            return copy.deepcopy(result)

    def request(self, identifier):
        with self.lock:
            return copy.deepcopy(self.requests.get(identifier))

    def boundary(self, save, validate, config, store, step):
        """唯一消费命令的位置；此时没有优化器更新或数据预取任务运行。"""
        while True:
            if self.stop_event.is_set():
                return False
            waiting = not self.running
            wait_started = time.monotonic()
            try:
                identifier, name, value = self.queue.get(timeout=.1 if not self.running else 0)
            except queue.Empty:
                if self.running:
                    self.publish(state="training", message="正在使用固定离线数据更新 IQL")
                    return True
                self.publish(state="paused", message="已暂停，可保存、验证或修改运行参数")
                continue
            finally:
                # 只累计等待命令的时间，手动验证与保存由各自阶段计时。
                if waiting:
                    self.timing("paused", time.monotonic()-wait_started)
            try:
                if name == "resume":
                    self.running = True
                elif name == "pause":
                    self.running = False
                elif name == "save":
                    save(snapshot=True)
                elif name == "validate":
                    if self.running:
                        raise ValueError("验证要求暂停")
                    validate()
                elif name == "apply":
                    if self.running:
                        raise ValueError("修改参数要求暂停")
                    self.apply(value, config, store, step)
                    save()
                with self.lock:
                    self.requests[identifier].update(status="done", message="已执行")
            except Exception as error:
                with self.lock:
                    self.requests[identifier].update(status="failed", message=str(error))

    def apply(self, value, config, store, step):
        changes = value.get("changes", {})
        if not isinstance(changes, dict) or not changes or set(changes) - EDITABLE.keys():
            raise ValueError("仅允许修改表单列出的运行参数")
        updated = copy.deepcopy(config)
        before = {}
        for key, candidate in changes.items():
            low, high, kind, _ = EDITABLE[key]
            if type(candidate) not in ((int,) if kind == "int" else (int, float)) or not low <= candidate <= high:
                raise ValueError(f"{key} 超出范围或类型不正确")
            section, field = key.split(".")
            before[key] = updated[section][field]
            updated[section][field] = candidate
        if updated["training"]["total_steps"] <= step:
            raise ValueError("总步数必须大于当前步数")
        atomic_json(self.output / "runtime_config.json", {k: updated[k] for k in ("data", "training", "iql", "reward", "output")})
        config.clear()
        config.update(updated)
        import torch
        torch.set_num_threads(config["training"]["cpu_threads"])
        with store.lock:
            store.budget = int(config["data"]["cache_gb"] * 1024**3)
            while store.cache and store.bytes > store.budget:
                _, old = store.cache.popitem(last=False)
                store.bytes -= sum(array.nbytes for array in old.values())
        self.stage += 1
        row = dict(step=step, stage=self.stage, before=before, changes=changes, time=time.time())
        with (self.output / "configuration.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")
        atomic_json(self.output / "config.json", config)
        self.record("configuration", row)
        self.publish(config=config, stage=self.stage)

    def _run(self):
        from .runtime import run
        try:
            run(self.config_path, self.init_bc, self.resume_path, control=self)
            self.publish(state="stopped" if self.stop_event.is_set() else "completed", message="训练已结束，模型已保存")
        except InterruptedError:
            self.publish(state="stopped", message="已取消数据准备；没有保存未初始化模型")
        except Exception as error:
            logging.exception("IQL 工作台训练失败")
            self.publish(state="error", error=str(error), message="训练线程已停止，请检查错误并使用最后完整保存的模型")
        finally:
            with self.lock:
                for result in self.requests.values():
                    if result["status"] == "queued":
                        ok = self.status["state"] == "stopped" and result.get("command") == "stop"
                        result.update(status="done" if ok else "failed", message=self.status["message"])

    def close(self):
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join()
