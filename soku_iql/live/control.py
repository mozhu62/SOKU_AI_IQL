from __future__ import annotations

import ctypes
import threading
import time
from ctypes import wintypes

from ..bc_core.schema import BUTTON_FEATURES
from ..bc_core.action_space import to_controller, validate_buttons
from .windows_api import WindowsApi, EmergencyPauseKey, parse_virtual_key, virtual_key_name


class SafeControl:
    """后台看门狗独立于模型推理，失焦、F10、超时均释放本控制器的按键。"""

    def __init__(self, config):
        self.config = config
        self.api = WindowsApi()
        self.keys = {name: parse_virtual_key(key) for name, key in config["keyboard"].items()}
        if len(set(self.keys.values())) != len(self.keys) or 0x79 in self.keys.values():
            raise ValueError("八个游戏按键必须互不重复，且不能占用 F10")
        self.confirm = parse_virtual_key(config["restart"]["confirm_key"])
        if self.confirm == 0x79:
            raise ValueError("续局键不能使用紧急暂停键 F10")
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
        self.kernel.CreateMutexW.restype = wintypes.HANDLE
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.kernel.CloseHandle.restype = wintypes.BOOL
        self.kernel.GetTickCount64.restype = ctypes.c_uint64
        # 兼容现有 PPO 的互斥名称，阻止两个控制器同时往同一桌面发键；不导入 PPO。
        self.mutex = self.kernel.CreateMutexW(None, False, "Local\\SokuPpo.KeyboardController.v1")
        error = ctypes.get_last_error()
        if not self.mutex:
            raise OSError(error, "无法建立实战控制互斥锁")
        if error == 183:
            self.kernel.CloseHandle(self.mutex)
            self.mutex = None
            raise RuntimeError("已有 IQL/BC/CQL/PPO 控制器运行，请先关闭；DQN 控制器也不能同时使用")
        self.lock = threading.RLock()
        self.closed = threading.Event()
        self.active = False
        self.focus_seen = False
        self.ready_at = 0.0
        self.pid = 0
        self.revision = 0
        self.pressed = set()
        self.heartbeat = time.monotonic()
        self.reason = "模型加载后点击开始，或先进入游戏再开始"
        self.emergency = EmergencyPauseKey(self.api, 0x79)
        self.thread = threading.Thread(target=self._watch, name="bc-input-watchdog", daemon=True)
        self.thread.start()

    def publisher_age_ms(self, payload):
        return max(0, int(self.kernel.GetTickCount64()) - int(payload.publisherTickMs))

    def touch(self):
        with self.lock:
            self.heartbeat = time.monotonic()

    def resume(self, pid):
        with self.lock:
            if not pid:
                self.reason = "尚未连接游戏，请先启动载入 SokuDataBridge 的游戏"
                return
            self.pid = int(pid)
            self.active = True
            self.focus_seen = False
            self.touch()
            self.reason = "请切回游戏；Windows 拒绝自动聚焦时可手动 Alt+Tab"
            self.api.focus_process_window(self.pid)

    def _ready(self):
        if not self.active or self.api.foreground_process_id() != self.pid:
            return False
        if not self.focus_seen:
            self.focus_seen = True
            self.ready_at = time.monotonic() + self.config["environment"]["focus_delay_seconds"]
        return time.monotonic() >= self.ready_at

    def ready(self):
        with self.lock:
            return self._ready()

    def _set_keys(self, wanted):
        events = [(key, False) for key in sorted(self.pressed - wanted)]
        events += [(key, True) for key in sorted(wanted - self.pressed)]
        # 先登记可能已发送的键，SendInput 部分成功时也能在异常分支全部释放。
        self.pressed |= wanted
        self.api.send_key_events(events)
        self.pressed = set(wanted)

    def release(self):
        with self.lock:
            try:
                self._set_keys(set())
            except OSError:
                try:
                    self.api.send_key_events([(key, False) for key in sorted(self.pressed)])
                    self.pressed.clear()
                except OSError:
                    self.active = False
                    self.reason = "释放按键失败，请手动检查键盘和游戏/脚本权限"

    def pause(self, reason):
        with self.lock:
            self.active = False
            self.focus_seen = False
            self.reason = reason
            self.revision += 1
            self.release()

    def apply(self, direction, buttons):
        with self.lock:
            if not self._ready():
                return False
            if direction not in range(1, 10) or len(buttons) != 4 or any(x not in (0, 1) for x in buttons):
                raise ValueError("实战动作必须是九宫格方向和四个二值按钮")
            validate_buttons(buttons, "实际发键")
            # 方向是绝对屏幕坐标，不随朝向翻转；持续相同动作不会重复点按。
            wanted = set()
            if direction in (1, 4, 7):
                wanted.add(self.keys["left"])
            if direction in (3, 6, 9):
                wanted.add(self.keys["right"])
            if direction in (7, 8, 9):
                wanted.add(self.keys["up"])
            if direction in (1, 2, 3):
                wanted.add(self.keys["down"])
            wanted.update(self.keys[name] for name, held in zip(BUTTON_FEATURES, buttons) if held)
            try:
                self._set_keys(wanted)
            except OSError:
                self.pause("SendInput 失败，已暂停；请确认与游戏具有相同权限")
                raise
            return True

    def apply_joint(self, joint_action_id):
        direction, buttons = to_controller(joint_action_id)
        return self.apply(direction, buttons)

    def menu(self, pressed):
        with self.lock:
            if pressed and not self._ready():
                return False
            self._set_keys({self.confirm} if pressed else set())
            return True

    def snapshot(self):
        with self.lock:
            return {"active": self.active, "ready": self._ready(), "reason": self.reason,
                    "revision": self.revision,
                    "pressed_inputs": [name for name, key in self.keys.items() if key in self.pressed],
                    "pressed_keys": [virtual_key_name(key) for key in sorted(self.pressed)]}

    def _watch(self):
        while not self.closed.wait(0.02):
            try:
                with self.lock:
                    if self.emergency.poll_pressed_edge():
                        self.pause("F10 紧急暂停；恢复需点击开始")
                    elif self.active:
                        if time.monotonic() - self.heartbeat > self.config["environment"]["watchdog_seconds"]:
                            self.pause("推理/采样线程无响应，看门狗已释放按键")
                        elif self.focus_seen and self.api.foreground_process_id() != self.pid:
                            self.pause("游戏失焦，已松键；切回游戏后仍需点击开始")
                        else:
                            self._ready()
            except Exception as exc:
                self.pause(f"输入看门狗异常：{exc}")

    def close(self):
        self.closed.set()
        self.pause("已停止实战控制")
        self.thread.join(timeout=1)
        if self.mutex:
            self.kernel.CloseHandle(self.mutex)
            self.mutex = None


class MenuRestart:
    def __init__(self, config):
        self.config = config
        self.started = None
        self.last_tap = 0.0
        self.release_at = None
        self.presses = 0

    def reset(self, control):
        if self.release_at is not None:
            control.menu(False)
        self.started = self.release_at = None
        self.presses = 0

    def tick(self, control):
        if not self.config["enabled"] or not control.ready():
            return "整场结束，等待手动按 Z 续局"
        now = time.monotonic()
        if self.started is None:
            self.started = now
        if self.release_at is not None and now >= self.release_at:
            control.menu(False)
            self.release_at = None
        if now - self.started > self.config["timeout_seconds"] or self.presses >= self.config["max_presses"]:
            control.pause("自动续局超时/达到次数上限，请检查结算页面后再开始")
            return control.reason
        if self.release_at is None and now - self.last_tap >= self.config["interval_seconds"]:
            if control.menu(True):
                self.release_at = now + self.config["hold_seconds"]
                self.last_tap = now
                self.presses += 1
        return f"整场结束，连续点按 {self.config['confirm_key']}：{self.presses} 次"
