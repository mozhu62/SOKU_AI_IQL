from __future__ import annotations

import copy
import queue
import threading
import time
import traceback
from concurrent.futures import Future

from ..config import resolve
from ..bc_core.action_space import action_catalog
from .agent import LiveAgent
from .control import SafeControl, MenuRestart
from .frame_stream import LiveFrameClient
from .tcn_runtime import run_tcn_loop
from .statistics import EvaluationStatistics, frame_key, readback
from .battle_state import battle_error, mode_details


class LiveRuntime:
    """只有工作线程读取模型/统计；界面通过命令队列和不可变副本交互。"""

    def __init__(self, config, prepared_agent=None):
        self.config = copy.deepcopy(config)
        self.commands = queue.Queue()
        self.closed = threading.Event()
        self.lock = threading.Lock()
        self.status = {"phase": "等待加载", "message": "点击加载模型后开始", "loaded": False}
        self.thread = None
        self.control = None
        self.agent = None
        self.prepared_agent = prepared_agent
        self.stats = None
        self.latest = None
        self.last_prediction = None
        self.last_inferred_key = None
        self.pending = None
        self.acknowledged = self.unconfirmed = self.stale_predictions = self.missing_frames = 0
        self.last_publish = 0.0
        self.match_finished = False
        self.control_revision = 0
        self.message = "等待游戏"
        self.last_traceback = None
        self.temporal_resets = 0
        self.temporal_reset_reason = None
        self.frame_queue_connected = False
        self.frame_queue_received = self.frame_queue_dropped = self.tcn_warmup_waits = 0

    def start(self):
        if self.thread is not None:
            raise RuntimeError("该实战会话已经启动；换模型请先停止")
        self.thread = threading.Thread(target=self._run, name="bc-live-evaluation", daemon=False)
        self.thread.start()

    def command(self, name):
        if name not in ("resume", "pause", "stop", "save"):
            raise ValueError(f"未知实战命令：{name}")
        completed = Future()
        if name == "stop":
            self.closed.set()
            completed.set_result("已请求停止并松键")
        elif self.closed.is_set():
            completed.set_exception(RuntimeError("实战会话已停止，请重新加载模型"))
        else:
            self.commands.put((name, completed))
        return completed

    def snapshot(self):
        with self.lock:
            return copy.deepcopy(self.status)

    def _publish(self, phase=None, force=False):
        now = time.monotonic()
        if not force and now - self.last_publish < 0.2:
            return
        self.last_publish = now
        status = {"phase": phase or ("运行中" if self.control and self.control.ready() else "等待/暂停"),
                  "message": self.message, "loaded": self.agent is not None,
                  "checkpoint": self.agent.path if self.agent else self.config["checkpoint"], "error": self.last_traceback,
                  "acknowledged": self.acknowledged, "unconfirmed": self.unconfirmed,
                  "stale_predictions": self.stale_predictions, "missing_frames": self.missing_frames,
                  "temporal_resets": self.temporal_resets, "temporal_reset_reason": self.temporal_reset_reason,
                  "prediction": copy.deepcopy(self.last_prediction)}
        if self.agent is not None:
            status.update(step=self.agent.step, device=self.agent.device_label,
                          inference_amp=self.agent.inference_amp, inference_dtype=self.agent.inference_dtype,
                          algorithm=self.agent.trained_algorithm, output_semantics="categorical_logits",
                          action_selection=self.agent.action_selection,
                          network_version=self.agent.model.spec["network_version"],
                          temporal=self.agent.model.spec["temporal"],
                          uses_resources=self.agent.model.uses_resources, action_catalog=action_catalog())
            window = self.agent.tcn_window
            status["temporal_capture"] = {
                "source": "LiveFrames.v1", "connected": self.frame_queue_connected, "capacity": 128,
                "required_frames": window.size, "ready": len(window) == window.size, "buffer_frames": len(window),
                "first_frame": window.rows[0][0][2] if len(window) else None,
                "last_frame": window.key[2] if window.key else None,
                "received_slots": self.frame_queue_received, "dropped_slots": self.frame_queue_dropped,
                "warmup_waits": self.tcn_warmup_waits, "padding_frames": 0,
            }
        if self.control is not None:
            status["control"] = self.control.snapshot()
        if self.stats is not None:
            status.update(self.stats.snapshot())
        if self.latest is not None:
            p = self.latest
            player, opponent = (p.left, p.right) if self.config["environment"]["player_side"] == "left" else (p.right, p.left)
            status["game"] = {"frame": int(p.battleFrame), "round": int(p.currentRound),
                              "pid": int(p.gameProcessId), "scene": int(p.sceneId),
                              "match_state": int(p.matchState), "in_battle": bool(p.inBattle),
                              "hp": int(player.hp), "opponent_hp": int(opponent.hp),
                              "actual_action": int(player.action), "action_frame": int(player.actionFrameCount),
                              "readback": readback(player, self.agent.vertical_positive_is_down) if self.agent else None}
            status["game"].update(mode_details(p))
        with self.lock:
            self.status = status

    def _reset_memory(self, reason="暂停、边界或观测失效"):
        if self.agent.memory is not None or len(getattr(self.agent, "tcn_window", None) or ()):
            self.temporal_resets += 1
            self.temporal_reset_reason = reason
        self.agent.reset()
        self.last_inferred_key = None
        self.pending = None

    def _handle_commands(self):
        while True:
            try:
                command, completed = self.commands.get_nowait()
            except queue.Empty:
                return
            if command == "resume":
                if self.stats.target and self.stats.completed >= self.stats.target:
                    self.message = "本次评估已完成；停止后重新加载可开始新会话"
                    completed.set_exception(ValueError(self.message))
                    continue
                pid = int(self.latest.gameProcessId) if self.latest is not None else 0
                self.control.resume(pid)
                self.message = self.control.reason
            elif command == "pause":
                self.control.pause("用户暂停；游戏本身不会暂停")
            elif command == "save":
                self.stats.save()
                self.message = f"评估汇总已保存：{self.stats.directory}"
            completed.set_result(self.control.reason if command == "pause" else self.message)

    def _active_battle(self, p):
        return bool(p.inBattle and not battle_error(p)
                    and int(p.matchState) in self.config["environment"]["active_match_states"]
                    and p.left.hp > 0 and p.right.hp > 0)

    def _observe(self, payload):
        old = self.latest
        self.latest = payload
        error = battle_error(payload)
        if error:
            self.control.pause(error)
            self.stats.cut(error)
            self._reset_memory(error)
            self.match_finished = False
            self.message = error
            return False
        if old is not None and (payload.gameProcessId != old.gameProcessId or not payload.inBattle):
            self._reset_memory("游戏进程变化或退出战斗")
        if old is not None and payload.gameProcessId != old.gameProcessId:
            self.control.pause("游戏进程已变化，请确认后重新开始")
            self.match_finished = False
            self.stats.cut("游戏进程变化")
        if old is not None and payload.inBattle and old.inBattle:
            gap = int(payload.battleFrame) - int(old.battleFrame)
            if frame_key(payload)[:2] == frame_key(old)[:2] and gap > 1:
                self.missing_frames += gap - 1
                self._reset_memory(f"缺失 {gap - 1} 个游戏帧，TCN 重新积累连续窗口")
                if gap > self.config["environment"]["max_memory_gap_frames"]:
                    self.stats.mark_partial("长时间缺帧，伤害统计不完整")
                    self._reset_memory("长时间缺失游戏帧")
            if gap < 0 or payload.currentRound != old.currentRound:
                self._reset_memory("帧号回退或小局变化")
        active = self._active_battle(payload)
        ready = self.control.ready()
        if active and ready:
            # 每个已观察的游戏帧都更新实际输入历史，而不只记录成功发送的模型指令。
            self.agent.builder.observe_inputs(payload)
        self.stats.observe(payload, ready, active)
        if (self.pending and frame_key(payload)[:2] == self.pending["key"][:2]
                and payload.sampleSerial > self.pending["serial"] and payload.battleFrame > self.pending["key"][2]):
            actual = readback(self.agent.builder.players(payload)[0], self.agent.vertical_positive_is_down)
            if actual == self.pending["action"]:
                self.acknowledged += 1
                self.pending = None
            elif payload.battleFrame - self.pending["key"][2] >= 8:
                self.unconfirmed += 1
                self.pending = None
        if active:
            self.match_finished = False
        elif payload.inBattle and (payload.left.score >= 2 or payload.right.score >= 2):
            # 第一次倒地可能先于 score 加一；以整场分数确认后才允许自动点 Z。
            self.match_finished = True
        if self.stats.target and self.stats.completed >= self.stats.target and self.control.snapshot()["active"]:
            self.control.pause(f"已完成 {self.stats.target} 个完整小局，评估停止且已保存")
        return active

    def _run(self):
        client = restart = None
        try:
            self.message = "加载 IQL/BC 策略与 checkpoint 内归一化参数"
            self._publish("加载中", True)
            self.control = SafeControl(self.config)
            # 工作台已在暂停期间校验并加载的新模型直接交接，不重复读取大文件。
            self.agent = self.prepared_agent or LiveAgent(resolve(self.config["checkpoint"]), self.config)
            self.prepared_agent = None
            self.agent.reset()
            self.stats = EvaluationStatistics(self.config, self.agent)
            client = LiveFrameClient()
            restart = MenuRestart(self.config["restart"])
            env = self.config["environment"]
            last_serial, last_frame, last_frame_time = None, None, time.monotonic()
            side = "1P" if env["player_side"] == "left" else "2P"
            self.message = f"模型就绪。进入对战后点击开始，当前控制 {side}，不限制双方角色"
            self._publish("就绪", True)
            run_tcn_loop(self, client, restart)
            return
            while not self.closed.wait(env["poll_seconds"]):
                self.control.touch()
                self._handle_commands()
                ctrl = self.control.snapshot()
                if ctrl["revision"] != self.control_revision:
                    self.control_revision = ctrl["revision"]
                    self.stats.cut(ctrl["reason"])
                    self._reset_memory()
                    restart.reset(self.control)
                snapshot = client.read()
                if snapshot is None or not snapshot.payload.initialized:
                    self.control.release()
                    self.message = client.wait_reason
                    if time.monotonic() - last_frame_time > env["stale_timeout_seconds"]:
                        client.close()
                        if ctrl["active"]:
                            self.control.pause(self.message)
                        self.stats.cut("游戏数据连接中断")
                        self._reset_memory()
                    self._publish()
                    continue
                p = snapshot.payload
                age = self.control.publisher_age_ms(p)
                if age > env["max_snapshot_age_ms"]:
                    self.control.release()
                    self._reset_memory()
                    self.stats.mark_partial("观测过期导致控制中断")
                    self.message = f"观测过期 {age} ms，等待新帧"
                    if age > env["stale_timeout_seconds"] * 1000:
                        self.control.pause("DLL 长时间未发布新观测，请检查游戏后继续")
                        client.close()
                    self._publish()
                    continue
                serial = int(p.gameProcessId), int(p.sampleSerial)
                if serial == last_serial:
                    if self._active_battle(p) and time.monotonic() - last_frame_time > env["stale_timeout_seconds"]:
                        self.control.release()
                        self.message = "等待新的游戏帧，当前未持续发键"
                    self._publish()
                    continue
                last_serial = serial
                # 相同战斗帧可以发生场景/终局变化，但不能重复统计或推进时序状态。
                marker = (*frame_key(p), int(p.inBattle), int(p.matchState), int(p.left.score), int(p.right.score),
                          int(p.left.hp), int(p.right.hp), int(p.battleMode), int(p.battleSubMode),
                          int(p.left.characterId), int(p.right.characterId))
                if marker != last_frame:
                    last_frame_time = time.monotonic()
                    last_frame = marker
                    active = self._observe(p)
                else:
                    active = self._active_battle(p)
                    if active and time.monotonic() - last_frame_time > env["stale_timeout_seconds"]:
                        self.control.release()
                        self._reset_memory()
                        self.stats.mark_partial("战斗帧停止推进")
                        self.message = "游戏帧停住，已松键，等待继续"
                        self._publish()
                        continue
                if not self.control.ready():
                    self.message = self.control.reason
                    self._publish()
                    continue
                if not active:
                    self._reset_memory()
                    if self.match_finished:
                        self.message = restart.tick(self.control)
                    else:
                        self.control.release()
                        self.message = f"等待战斗：scene={p.sceneId}, matchState={p.matchState}, inBattle={p.inBattle}"
                    self._publish()
                    continue
                restart.reset(self.control)
                key = frame_key(p)
                if self.last_inferred_key and key[:2] == self.last_inferred_key[:2]:
                    if 0 <= key[2] - self.last_inferred_key[2] < env["decision_interval_frames"]:
                        self._publish()
                        continue
                prediction, next_memory = self.agent.predict(p, snapshot.resources)
                # 预测与实际发键分别展示：即使观测过期而没有执行，也保留本次网络输出供排查。
                prediction.update(execution_status="pending", execution_reason="等待发键前安全检查")
                self.last_prediction = prediction
                # 推理完成后再查前台、终局和观测年龄，禁止把旧局动作发到菜单/下一局。
                fresh = client.read_state()
                valid = fresh is not None and fresh.payload.initialized
                if valid:
                    latest = fresh.payload
                    lag = int(latest.battleFrame) - int(p.battleFrame)
                    valid = (frame_key(latest)[:2] == key[:2] and self._active_battle(latest)
                             and 0 <= lag <= env["max_inference_lag_frames"]
                             and self.control.publisher_age_ms(latest) <= env["max_snapshot_age_ms"])
                if self.closed.is_set():
                    prediction.update(execution_status="not_sent", execution_reason="会话已停止，未发送")
                    break
                if not valid:
                    prediction.update(execution_status="discarded", execution_reason="推理后观测失效，未发送")
                    self.stale_predictions += 1
                    self.control.release()
                    self._reset_memory()
                    self.stats.mark_partial("推理后观测失效，未执行模型动作")
                    self.message = "推理期间观测已变化，丢弃旧动作并追最新帧"
                    self._publish()
                    continue
                if self.control.apply_joint(prediction["joint_action_id"]):
                    prediction.update(execution_status="sent", execution_reason="已发送，游戏实际输入请看回读")
                    self.agent.memory = next_memory
                    self.last_inferred_key = key
                    self.last_prediction = prediction
                    self.stats.decision(prediction)
                    if self.pending:
                        self.unconfirmed += 1
                    self.pending = {"key": key, "serial": int(latest.sampleSerial),
                                    "action": (prediction["direction"], prediction["buttons"])}
                    self.message = "固定模型实战中；仅推理与统计，不训练、不覆盖 checkpoint"
                else:
                    prediction.update(execution_status="not_sent", execution_reason="控制未就绪或游戏失焦，未发送")
                    self._reset_memory()
                self._publish()
        except Exception as exc:
            self.last_traceback = traceback.format_exc()
            self.message = f"实战已安全停止：{exc}"
            if self.last_prediction and self.last_prediction.get("execution_status") == "pending":
                self.last_prediction.update(execution_status="error", execution_reason="执行发生异常，已请求松键")
            if self.control:
                self.control.pause(self.message)
            self._publish("错误", True)
        finally:
            # 松键优先于任何磁盘写入，即使报告目录写入失败也不能卡住角色。
            if self.control:
                self.control.close()
            if client:
                client.close()
            try:
                if self.stats:
                    self.stats.cut("评估会话停止")
                    self.stats.save()
            except Exception as exc:
                self.message += f"；保存报告失败：{exc}"
            self._publish("错误" if self.last_traceback else "已停止", True)
            self.closed.set()
            while not self.commands.empty():
                _, completed = self.commands.get_nowait()
                if not completed.done():
                    completed.set_exception(RuntimeError(self.message))
