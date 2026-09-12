from __future__ import annotations

import ctypes
import os
import threading
from pathlib import Path


class ModelPicker:
    """只允许用户通过 Windows 原生文件窗口授权一个文件，不开放网页任意路径读取。"""

    def __init__(self):
        self.cancelled = threading.Event()
        self.lock = threading.Lock()
        self.window = None
        self.user32 = None

    def prepare(self):
        self.cancelled.clear()

    def cancel(self):
        self.cancelled.set()
        with self.lock:
            if self.window and self.user32:
                self.user32.PostMessageW(self.window, 0x0111, 2, 0)  # WM_COMMAND / IDCANCEL

    def choose(self, initial: Path) -> Path | None:
        if os.name != "nt":
            raise ValueError("本机文件选择需要 Windows 桌面；实战服务应在游戏所在电脑运行")
        if self.cancelled.is_set():
            return None
        from ctypes import wintypes as w

        # 使用指针宽度的回调/参数，兼容 32 位及 64 位 Python，路径始终使用 Unicode。
        hook_type = ctypes.WINFUNCTYPE(ctypes.c_size_t, w.HWND, w.UINT, w.WPARAM, w.LPARAM)

        class OpenFileName(ctypes.Structure):
            _fields_ = [
                ("lStructSize", w.DWORD), ("hwndOwner", w.HWND), ("hInstance", w.HINSTANCE),
                ("lpstrFilter", w.LPCWSTR), ("lpstrCustomFilter", w.LPWSTR),
                ("nMaxCustFilter", w.DWORD), ("nFilterIndex", w.DWORD),
                ("lpstrFile", w.LPWSTR), ("nMaxFile", w.DWORD),
                ("lpstrFileTitle", w.LPWSTR), ("nMaxFileTitle", w.DWORD),
                ("lpstrInitialDir", w.LPCWSTR), ("lpstrTitle", w.LPCWSTR), ("Flags", w.DWORD),
                ("nFileOffset", w.WORD), ("nFileExtension", w.WORD), ("lpstrDefExt", w.LPCWSTR),
                ("lCustData", w.LPARAM), ("lpfnHook", hook_type), ("lpTemplateName", w.LPCWSTR),
                ("pvReserved", ctypes.c_void_p), ("dwReserved", w.DWORD), ("FlagsEx", w.DWORD),
            ]

        user32 = ctypes.WinDLL("user32", use_last_error=True)
        user32.GetParent.argtypes, user32.GetParent.restype = [w.HWND], w.HWND
        user32.PostMessageW.argtypes = [w.HWND, w.UINT, w.WPARAM, w.LPARAM]
        user32.PostMessageW.restype = w.BOOL
        user32.SetForegroundWindow.argtypes, user32.SetForegroundWindow.restype = [w.HWND], w.BOOL
        common = ctypes.WinDLL("comdlg32", use_last_error=True)
        common.GetOpenFileNameW.argtypes, common.GetOpenFileNameW.restype = [ctypes.POINTER(OpenFileName)], w.BOOL
        common.CommDlgExtendedError.argtypes, common.CommDlgExtendedError.restype = [], w.DWORD

        @hook_type
        def hook(child, message, _wparam, _lparam):
            if message == 0x0110:  # WM_INITDIALOG；Explorer 回调给的是子窗口，需要取得父窗口。
                window = user32.GetParent(child)
                with self.lock:
                    self.window, self.user32 = window, user32
                if self.cancelled.is_set():
                    user32.PostMessageW(window, 0x0111, 2, 0)
                else:
                    user32.SetForegroundWindow(window)
            return 0

        buffer = ctypes.create_unicode_buffer(32768)
        options = OpenFileName()
        options.lStructSize = ctypes.sizeof(options)
        options.lpstrFilter = "IQL/BC 策略 (*.pt)\0*.pt\0\0"
        options.nFilterIndex = 1
        options.lpstrFile = ctypes.cast(buffer, w.LPWSTR)
        options.nMaxFile = len(buffer)
        options.lpstrInitialDir = str(initial if initial.is_dir() else initial.parent)
        options.lpstrTitle = "选择 IQL 实战模型 — 选择后请在网页点击加载"
        options.lpstrDefExt = "pt"
        # 文件必须存在；允许调整窗口大小，不写最近文档，也不创建或复制模型文件。
        options.Flags = 0x00080000 | 0x00001000 | 0x00000800 | 0x00000008 | 0x00000004 | 0x02000000 | 0x00000020 | 0x00800000
        options.lpfnHook = hook
        try:
            accepted = common.GetOpenFileNameW(ctypes.byref(options))
            if not accepted:
                error = common.CommDlgExtendedError()
                if error:
                    raise OSError(f"Windows 文件选择窗口失败，错误码 0x{error:08X}")
                return None
            return None if self.cancelled.is_set() else Path(buffer.value)
        finally:
            with self.lock:
                self.window, self.user32 = None, None
