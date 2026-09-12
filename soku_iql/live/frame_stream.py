from __future__ import annotations

import ctypes as ct
from dataclasses import dataclass

from .shared_state import SharedMemoryClient, StatePayload, FILE_MAP_READ, SharedMemoryProtocolError
from .resource_state import Resources, ResourceFrame


CAPACITY = 128
MAGIC = 0x534C4631
VERSION = 1


class FrameHeader(ct.Structure):
    _fields_ = [(name, ct.c_uint32) for name in (
        "magic", "protocolVersion", "structureSize", "payloadSize", "resourceSize",
        "capacity", "slotSize", "gameProcessId", "writeSequence", "reserved")]
    _fields_ += [("latestSerial", ct.c_uint64), ("generation", ct.c_uint64)]


class FrameSlot(ct.Structure):
    _fields_ = [("writeSequence", ct.c_uint32), ("reserved", ct.c_uint32), ("serial", ct.c_uint64),
                ("payload", StatePayload), ("resources", Resources)]


MAPPING_BYTES = ct.sizeof(FrameHeader) + CAPACITY * ct.sizeof(FrameSlot)


def validate_header(header, pid):
    expected = (MAGIC, VERSION, MAPPING_BYTES, ct.sizeof(StatePayload), ct.sizeof(Resources),
                CAPACITY, ct.sizeof(FrameSlot), pid)
    actual = tuple(int(getattr(header, name)) for name in (
        "magic", "protocolVersion", "structureSize", "payloadSize", "resourceSize",
        "capacity", "slotSize", "gameProcessId"))
    if (ct.sizeof(FrameHeader) != 56 or FrameSlot.payload.offset != 16
            or FrameSlot.resources.offset != 16 + ct.sizeof(StatePayload) or actual != expected):
        raise SharedMemoryProtocolError("LiveFrames.v1 帧队列协议不匹配，请更新采集 DLL；禁止降级为稀疏历史")


def unread_range(cursor, latest):
    first = max(cursor + 1, latest - CAPACITY + 1)
    return range(first, latest + 1), max(0, first - cursor - 1)


@dataclass(frozen=True)
class CapturedFrame:
    payload: StatePayload
    resources: ResourceFrame


def captured_frame(slot):
    p = slot.payload
    resource = ResourceFrame()
    for key in ("sampleSerial", "publisherTickMs", "gameProcessId", "battleFrame", "currentRound", "initialized"):
        setattr(resource, key, getattr(p, key))
    resource.resources = slot.resources
    return CapturedFrame(p, resource)


class LiveFrameClient:
    """按游标读真实逐帧队列；槽内包含同帧技能，不依赖只保留四帧的 Resources 通道。"""

    def __init__(self):
        self.state = SharedMemoryClient()
        self.kernel = self.state._kernel32
        self.mapping = self.address = self.pid = None
        self.cursor = 0
        self.generation = None
        self.reset_reason = None
        self.available = False
        self.wait_reason = "等待 LiveFrames.v1 真实帧队列"

    def read_state(self):
        # 这里只用于发键前复核最新场景/年龄，不推进历史游标。
        return self.state.read()

    def _header(self):
        address = self.address + FrameHeader.writeSequence.offset
        for _ in range(4):
            before = ct.c_uint32.from_address(address).value
            if before & 1:
                continue
            header = FrameHeader.from_buffer_copy(ct.string_at(self.address, ct.sizeof(FrameHeader)))
            after = ct.c_uint32.from_address(address).value
            if before == after and not after & 1:
                if header.structureSize == 0:
                    return None
                validate_header(header, self.pid)
                return header
        return None

    def _connect(self, pid):
        if self.address is not None and pid == self.pid:
            return True
        previous_pid = self.pid
        self._close_ring()
        mapping = self.kernel.OpenFileMappingW(FILE_MAP_READ, False, f"Local\\SokuDataBridge.LiveFrames.v1.{pid}")
        if not mapping:
            self.wait_reason = "TCN32 需要 LiveFrames.v1 逐帧队列：请构建并替换新版 SokuDataBridge.dll 后重启游戏；不使用短历史发键"
            return False
        address = self.kernel.MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, MAPPING_BYTES)
        if not address:
            self.kernel.CloseHandle(mapping)
            raise SharedMemoryProtocolError("LiveFrames 帧队列映射失败，请检查 DLL 版本和结构大小")
        self.mapping, self.address, self.pid = mapping, int(address), pid
        if previous_pid is not None and previous_pid != pid:
            self.reset_reason = "游戏进程变化"
        return True

    def _slot(self, serial):
        address = self.address + ct.sizeof(FrameHeader) + ((serial - 1) % CAPACITY) * ct.sizeof(FrameSlot)
        for _ in range(4):
            before = ct.c_uint32.from_address(address).value
            if before & 1:
                continue
            slot = FrameSlot.from_buffer_copy(ct.string_at(address, ct.sizeof(FrameSlot)))
            after = ct.c_uint32.from_address(address).value
            if before == after and not after & 1 and slot.serial == serial:
                if slot.payload.gameProcessId != self.pid:
                    raise SharedMemoryProtocolError("帧队列槽的进程标识不匹配，已停止控制")
                return captured_frame(slot)
        return None

    def read(self):
        self.reset_reason = None
        self.available = False
        discovery = self.read_state()
        if discovery is None or not discovery.payload.initialized:
            self.wait_reason = "等待已加载新版 SokuDataBridge 的游戏"
            return [], 0
        if not self._connect(int(discovery.payload.gameProcessId)):
            return [], 0
        header = self._header()
        if header is None:
            self.wait_reason = "帧队列正在发布，等待完整提交"
            return [], 0
        self.available = True
        latest = int(header.latestSerial)
        if self.generation != header.generation or latest < self.cursor:
            if self.generation is not None:
                self.reset_reason = "DLL 帧队列重新初始化"
            self.generation = int(header.generation)
            # 初次连接可追读已有真实历史；实际回合边界仍由观测层切断。
            self.cursor = max(0, latest - CAPACITY)
        serials, dropped = unread_range(self.cursor, latest)
        frames = []
        for serial in serials:
            frame = self._slot(serial)
            if frame is None:
                dropped += 1
            else:
                frames.append(frame)
        after = self._header()
        if after is None or after.generation != header.generation:
            self.available = False
            self.wait_reason = "帧队列代次变化，等待重新读取"
            return [], 0
        self.cursor = latest
        self.wait_reason = "已连接状态＋同帧资源队列，容量 128 帧"
        return frames, dropped

    def _close_ring(self):
        if self.address:
            self.kernel.UnmapViewOfFile(self.address)
        if self.mapping:
            self.kernel.CloseHandle(self.mapping)
        self.address = self.mapping = None
        self.cursor = 0
        self.generation = None

    def close(self):
        self.available = False
        self._close_ring()
        self.state.close()
