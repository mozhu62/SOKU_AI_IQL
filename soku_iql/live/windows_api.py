from __future__ import annotations

# 复用旧 DQN 已采用的扫描码 SendInput 和按 PID 聚焦方式，只保留通用 Windows 层。

import ctypes
import sys
from ctypes import wintypes



KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008
INPUT_KEYBOARD = 1
MAPVK_VK_TO_VSC = 0
SW_RESTORE = 9


VK_ALIASES: dict[str, int] = {
    "LEFT": 0x25,
    "RIGHT": 0x27,
    "UP": 0x26,
    "DOWN": 0x28,
    "SPACE": 0x20,
    "ENTER": 0x0D,
    "TAB": 0x09,
    "SHIFT": 0x10,
    "CTRL": 0x11,
    "ALT": 0x12,
    "ESC": 0x1B,
    "F10": 0x79,
}
EXTENDED_KEYS = {0x25, 0x26, 0x27, 0x28}


def parse_virtual_key(value: str | int) -> int:
    if isinstance(value, int):
        if 0 <= value <= 0xFF:
            return value
        raise ValueError(f"虚拟键码超出范围: {value}")
    name = value.strip().upper()
    if name in VK_ALIASES:
        return VK_ALIASES[name]
    if len(name) == 1 and ("A" <= name <= "Z" or "0" <= name <= "9"):
        return ord(name)
    if name.startswith("0X"):
        return parse_virtual_key(int(name, 16))
    raise ValueError(f"不支持的按键名称: {value}")


def virtual_key_name(value: int) -> str:
    for name, code in VK_ALIASES.items():
        if code == int(value):
            return name
    if 0x30 <= int(value) <= 0x39 or 0x41 <= int(value) <= 0x5A:
        return chr(int(value))
    return f"0x{int(value):02X}"


ULONG_PTR = wintypes.WPARAM


class MouseInput(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class KeyboardInput(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class HardwareInput(ctypes.Structure):
    _fields_ = [
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    ]


class InputUnion(ctypes.Union):
    _fields_ = [
        ("mi", MouseInput),
        ("ki", KeyboardInput),
        ("hi", HardwareInput),
    ]


class Input(ctypes.Structure):
    _anonymous_ = ("union",)
    _fields_ = [
        ("type", wintypes.DWORD),
        ("union", InputUnion),
    ]


class WindowsApi:
    def __init__(self) -> None:
        if sys.platform != "win32":
            raise OSError("键盘控制只能在 Windows 上运行")
        self.user32 = ctypes.WinDLL("user32", use_last_error=True)
        self._configure_api()

    def _configure_api(self) -> None:
        self.user32.SendInput.argtypes = [
            wintypes.UINT,
            ctypes.POINTER(Input),
            ctypes.c_int,
        ]
        self.user32.SendInput.restype = wintypes.UINT
        self.user32.MapVirtualKeyW.argtypes = [wintypes.UINT, wintypes.UINT]
        self.user32.MapVirtualKeyW.restype = wintypes.UINT
        self.user32.GetForegroundWindow.restype = wintypes.HWND
        self.user32.GetWindowThreadProcessId.argtypes = [
            wintypes.HWND,
            ctypes.POINTER(wintypes.DWORD),
        ]
        self.user32.GetWindowThreadProcessId.restype = wintypes.DWORD
        self.user32.IsWindowVisible.argtypes = [wintypes.HWND]
        self.user32.IsWindowVisible.restype = wintypes.BOOL
        self.user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
        self.user32.ShowWindow.restype = wintypes.BOOL
        self.user32.BringWindowToTop.argtypes = [wintypes.HWND]
        self.user32.BringWindowToTop.restype = wintypes.BOOL
        self.user32.SetForegroundWindow.argtypes = [wintypes.HWND]
        self.user32.SetForegroundWindow.restype = wintypes.BOOL
        self.user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
        self.user32.GetAsyncKeyState.restype = ctypes.c_short

    def process_id_for_window(self, window: int | None) -> int:
        if not window:
            return 0
        process_id = wintypes.DWORD()
        self.user32.GetWindowThreadProcessId(window, ctypes.byref(process_id))
        return int(process_id.value)

    def foreground_process_id(self) -> int:
        return self.process_id_for_window(self.user32.GetForegroundWindow())

    def find_window_for_process(self, process_id: int) -> int | None:
        matches: list[int] = []
        callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

        @callback_type
        def visit(window: int, _: int) -> bool:
            if self.user32.IsWindowVisible(window) and (
                self.process_id_for_window(window) == int(process_id)
            ):
                matches.append(int(window))
                return False
            return True

        self.user32.EnumWindows(visit, 0)
        return matches[0] if matches else None

    def focus_process_window(self, process_id: int) -> bool:
        window = self.find_window_for_process(process_id)
        if window is None:
            return False
        self.user32.ShowWindow(window, SW_RESTORE)
        self.user32.BringWindowToTop(window)
        self.user32.SetForegroundWindow(window)
        return self.foreground_process_id() == int(process_id)

    def is_key_down(self, virtual_key: int) -> bool:
        return bool(self.user32.GetAsyncKeyState(int(virtual_key)) & 0x8000)

    def send_key_events(self, events: list[tuple[int, bool]]) -> None:
        if not events:
            return
        inputs = (Input * len(events))()
        for index, (virtual_key, pressed) in enumerate(events):
            scan_code = int(self.user32.MapVirtualKeyW(virtual_key, MAPVK_VK_TO_VSC))
            flags = KEYEVENTF_SCANCODE
            if virtual_key in EXTENDED_KEYS:
                flags |= KEYEVENTF_EXTENDEDKEY
            if not pressed:
                flags |= KEYEVENTF_KEYUP
            inputs[index].type = INPUT_KEYBOARD
            inputs[index].ki = KeyboardInput(
                0,
                scan_code,
                flags,
                0,
                0,
            )
        sent = int(self.user32.SendInput(len(events), inputs, ctypes.sizeof(Input)))
        if sent != len(events):
            error = ctypes.get_last_error()
            raise OSError(error, f"SendInput 仅发送 {sent}/{len(events)} 个事件")


class EmergencyPauseKey:
    def __init__(self, api: WindowsApi, virtual_key: int) -> None:
        self.api = api
        self.virtual_key = int(virtual_key)
        self._was_down = False

    def poll_pressed_edge(self) -> bool:
        is_down = self.api.is_key_down(self.virtual_key)
        pressed = is_down and not self._was_down
        self._was_down = is_down
        return pressed
