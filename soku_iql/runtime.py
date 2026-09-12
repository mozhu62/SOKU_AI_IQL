from __future__ import annotations

import copy
import json
import logging
import math
import signal
import threading
import time
from uuid import uuid4
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from filelock import FileLock

from . import checkpoint
from .bc_core.storage import atomic_json
from .config import load, validate_resume, ACTOR_SAMPLING, ACTOR_WEIGHTING
from .dataset import IQLReplayStore, fixed_split
from .learner import Learner

LOGGER = logging.getLogger(__name__)


def aggregate(rows):
    samples = sum(row["samples"] for row in rows)
    result = {key: sum(row[key] * row["samples"] for row in rows) / samples
              for key in ("nll", "top1", "top5", "td_mse", "td_mae", "q_mean", "v_mean")}
    for prefix in ("target", "residual"):
        mean = sum(row[f"{prefix}_sum"] for row in rows) / samples
        result[f"{prefix}_variance"] = max(0, sum(row[f"{prefix}_square_sum"] for row in rows) / samples - mean * mean)
    result["ev"] = (1 - result["residual_variance"] / result["target_variance"]
                    if result["target_variance"] > 1e-12 else None)
    changes = sum(row["change_count"] for row in rows)
    eligible = sum(row["previous_eligible"] for row in rows)
    result.update(samples=samples, change_count=changes,
                  change_top1=sum(row["change_correct"] for row in rows) / changes if changes else None,
                  previous_action_baseline=sum(row["previous_correct"] for row in rows) / eligible if eligible else None)
    return result


class Prefetch:
    def __init__(self, store, config, step):
        self.store, self.config, self.next_step = store, config, step
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="iql-data")
        self.pending = deque()
        for _ in range(config["training"]["prefetch_batches"]):
            self.submit()

    def submit(self):
        rng = np.random.default_rng(np.random.SeedSequence([self.config["seed"], self.next_step, 0]))
        self.next_step += 1
        self.pending.append(self.executor.submit(self.store.sample, rng, self.config["training"]))

    def get(self):
        batch = self.pending.popleft().result()
        self.submit()
        return batch

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)


def run(config_path, init_bc=None, resume=None, control=None):
    if bool(init_bc) == bool(resume):
        raise ValueError("首次训练必须 --init-bc；IQL 续训必须 --resume；二者不能同时使用")
    package, source_hash = checkpoint.load_source(init_bc or resume, bc=bool(init_bc))
    bc_config = copy.deepcopy(package["config"] if init_bc else package["bc_config"])
    config = load(config_path, bc_config)
    if resume:
        validate_resume(config, package["config"])
        if config["iql"]["n_step"] != package["config"]["iql"].get("n_step", 1):
            LOGGER.warning("TD 回报跨度变更：%s -> %s；保留权重和优化器，TD 误差不能与旧跨度直接比较",
                           package["config"]["iql"].get("n_step", 1), config["iql"]["n_step"])
        previous_sampling = package["config"].get("actor_sampling", ACTOR_SAMPLING)
        previous_weighting = package["config"].get("actor_weighting", ACTOR_WEIGHTING)
        if previous_weighting != config["actor_weighting"]:
            LOGGER.warning("Actor 类别加权变更：%s -> %s；保留模型和优化器，训练目标权重已改变",
                           previous_weighting, config["actor_weighting"])
        if previous_sampling != config["actor_sampling"]:
            LOGGER.warning("Actor 监督采样规则变更：%s -> %s；保留模型和优化器，但不是原分布的精确续训",
                           previous_sampling, config["actor_sampling"])
    output = Path(config["output"]["directory"])
    source_path = Path(init_bc or resume).resolve()
    if control:
        control.attach(config, source_path)
    managed = ("last.pt", "actor_bc.pt", "best_actor_bc.pt", "bc_initial_actor.pt")
    if source_path in tuple((output / name).resolve() for name in managed) and init_bc:
        raise ValueError("IQL 输出不得覆盖初始化 BC 文件")
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output / "training.lock"), timeout=0):
        if init_bc and any((output / name).exists() for name in managed):
            raise ValueError("输出已有 IQL 模型；请 --resume 或指定新目录")
        prepare_started = time.perf_counter()
        def progress(message):
            LOGGER.info(message)
            if control:
                control.progress(message)
        cancelled = control.stop_event.is_set if control else lambda: False
        split = fixed_split(config, progress, cancelled)
        if split["sha256"] != package["split_hash"]:
            raise ValueError("必须使用 BC 原固定训练/验证划分，不能混入新的验证数据")
        store = IQLReplayStore(config, split, progress, cancelled=cancelled, normalization=package["normalization"])
        torch.manual_seed(config["seed"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config["seed"])
        learner = Learner(config, package["model"] if init_bc else None)
        step = samples = actor_updates = 0
        best = float("inf")
        provenance = dict(bc_path=str(source_path), bc_sha256=source_hash)
        if resume:
            learner.networks.load_state_dict(package["networks"], strict=True)
            for key in learner.optimizers:
                learner.optimizers[key].load_state_dict(package["optimizers"][key])
                learner.scalers[key].load_state_dict(package["scalers"][key])
            torch.set_rng_state(package["rng_cpu"].cpu())
            if learner.device.type == "cuda" and package["rng_cuda"]:
                if len(package["rng_cuda"]) != torch.cuda.device_count():
                    raise ValueError("CUDA RNG 设备数不同，不能声称精确续训")
                torch.cuda.set_rng_state_all([state.cpu() for state in package["rng_cuda"]])
            step, samples, actor_updates, best = (package[key] for key in ("step", "samples", "actor_updates", "best_nll"))
            provenance = package["provenance"]
        else:
            if config["model"]["temporal_mode"] != package["config"]["model"]["temporal_mode"]:
                LOGGER.warning("Actor 时序扩展 %s→%s：复用全部旧层，新增卷积块随机初始化；输出不保证与原策略相同",
                               package["config"]["model"]["temporal_mode"], config["model"]["temporal_mode"])
            actor_state = learner.networks.actor.state_dict()
            if not all(torch.equal(actor_state[key].cpu(), tensor) for key, tensor in package["model"].items()):
                raise RuntimeError("BC→IQL Actor 权重未完整对齐")
        # 网络和优化器已接管状态，释放原包，避免整个训练周期额外保留一套 CPU 权重。
        del package
        if control:
            counts = [sum(store.info[name]["joint_counts"][i] for name in split["train"]) for i in range(144)]
            control.timing("preparation", time.perf_counter()-prepare_started)
            control.publish(step=step, samples=samples, actor_updates=actor_updates, device=str(learner.device),
                            data=dict(train_shards=len(split["train"]), validation_shards=len(split["validation"]),
                                      transitions=sum(row["transitions"] for row in store.info.values()),
                                      split_hash=split["sha256"], train_action_counts=counts),
                            best_nll=best if math.isfinite(best) else None)
        atomic_json(output / "config.json", config)
        atomic_json(output / "provenance.json", provenance)
        stop = False
        def request_stop(*_):
            nonlocal stop
            stop = True
            LOGGER.warning("收到停止请求，当前完整更新完成后保存退出")
        # Web 模式训练在后台线程，信号只由服务主线程处理。
        old_signal = signal.signal(signal.SIGINT, request_stop) if threading.current_thread() is threading.main_thread() else None
        prefetch = None
        def save(snapshot=False):
            started = time.perf_counter()
            checkpoint.save(output / "last.pt", learner, bc_config, store.normalization, split["sha256"],
                            step, samples, actor_updates, best, provenance)
            checkpoint.export_actor(output / "actor_bc.pt", learner, bc_config, store.normalization, split["sha256"],
                                    step, samples, actor_updates, provenance)
            if snapshot:
                checkpoint.save(output / "snapshots" / f"step_{step:09d}_{uuid4().hex[:8]}.pt", learner, bc_config,
                                store.normalization, split["sha256"], step, samples, actor_updates, best, provenance)
            if control:
                control.timing("save", time.perf_counter()-started)
                control.publish(last_saved=dict(step=step, time=time.time()))
        def record(kind, row):
            # AMP 溢出会跳过对应优化器；梯度诊断记 null，不写非法 JSON 的 Infinity。
            row = {key: (None if isinstance(value, float) and not math.isfinite(value) else value) for key, value in row.items()}
            row = dict(step=step, **row)
            if control:
                row["stage"] = control.stage
            with (output / f"{kind}.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            if control:
                control.record(kind, row)
        def validate():
            started = time.perf_counter()
            if control:
                control.publish(state="validating", message="固定种子离线验证，不更新参数")
            rows = []
            for index in range(config["training"]["validation_batches"]):
                rng = np.random.default_rng(np.random.SeedSequence([config["seed"], index, 1]))
                rows.append(learner.validate_batch(store.sample(rng, config["training"], "validation")))
            result = aggregate(rows)
            record("validation", result)
            LOGGER.info("验证 samples=%d NLL=%.4f Top1=%.3f TD_MSE=%.4f EV=%s change_Top1=%s",
                        result["samples"], result["nll"], result["top1"], result["td_mse"], result["ev"], result["change_top1"])
            if control:
                control.timing("validation", time.perf_counter()-started)
            return result
        def validate_and_rank():
            nonlocal best
            metrics = validate()
            if metrics["nll"] < best:
                best = metrics["nll"]
                checkpoint.export_actor(output / "best_actor_bc.pt", learner, bc_config, store.normalization,
                                        split["sha256"], step, samples, actor_updates, provenance)
                if control:
                    control.publish(best_nll=best)
            return metrics
        try:
            if init_bc:
                # 留存迁移前基线和可直接回放的策略；尚未进行任何策略更新。
                baseline = validate()
                best = baseline["nll"]
                atomic_json(output / "bc_baseline.json", baseline)
                checkpoint.export_actor(output / "bc_initial_actor.pt", learner, bc_config, store.normalization,
                                        split["sha256"], step, samples, actor_updates, provenance)
                save()
            if control:
                control.publish(ready=True, best_nll=best, state="paused" if not control.running else "training")
            while step < config["training"]["total_steps"] and not stop:
                if control:
                    # 暂停/改参/验证前排空旧预取；种子按 step 派生，丢弃预取不改变采样顺序。
                    if control.stop_event.is_set() or not control.queue.empty() or not control.running:
                        if prefetch is not None:
                            prefetch.close()
                            prefetch = None
                        if not control.boundary(save, validate_and_rank, config, store, step):
                            break
                    control.publish(state="training")
                if prefetch is None:
                    prefetch = Prefetch(store, config, step)
                started = time.perf_counter()
                batch = prefetch.get()
                wait = time.perf_counter() - started
                result = learner.train_batch(batch, step)
                step += 1
                samples += result["samples"]
                actor_updates += int(result["actor_updated"])
                if control:
                    control.update(step, samples, actor_updates, result, wait)
                if step % config["training"]["log_interval"] == 0:
                    result.update(data_wait_seconds=wait, steps_per_second=1 / max(wait + result["optimization_seconds"], 1e-9))
                    record("train", result)
                    LOGGER.info("step=%d IQL V=%.4f Q=%.4f actor=%s weight=%.3f speed=%.2f step/s warmup=%s",
                                step, result["value_loss"], result["q_loss"], result["actor_loss"], result["weight_mean"],
                                result["steps_per_second"], result["warmup"])
                if step % config["training"]["validation_interval"] == 0 and not stop and not cancelled():
                    prefetch.close()
                    prefetch = None
                    validate_and_rank()
                if step % config["training"]["save_interval"] == 0:
                    save()
            save()
        finally:
            if prefetch is not None:
                prefetch.close()
            if old_signal is not None:
                signal.signal(signal.SIGINT, old_signal)
