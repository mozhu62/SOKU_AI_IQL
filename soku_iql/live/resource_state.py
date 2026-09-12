from __future__ import annotations

import ctypes as ct

from .shared_state import SharedMemoryClient, SharedSnapshot


class SkillSlot(ct.Structure):
    _fields_ = [("variant", ct.c_int32), ("learnedLevel", ct.c_int32), ("effectiveLevel", ct.c_int32)]


class PlayerResources(ct.Structure):
    _fields_ = [("validMask", ct.c_uint32), ("skills", SkillSlot * 4)]


class Resources(ct.Structure):
    _fields_ = [("left", PlayerResources), ("right", PlayerResources)]


class ResourceFrame(ct.Structure):
    _fields_ = [("sampleSerial", ct.c_uint64), ("publisherTickMs", ct.c_uint64),
                ("gameProcessId", ct.c_uint32), ("battleFrame", ct.c_uint32),
                ("currentRound", ct.c_int32), ("initialized", ct.c_uint32), ("resources", Resources)]


class ResourceSharedState(ct.Structure):
    _fields_ = [("magic", ct.c_uint32), ("protocolVersion", ct.c_uint32),
                ("structureSize", ct.c_uint32), ("writeSequence", ct.c_int32), ("frames", ResourceFrame * 4)]


def matches(frame, state):
    # 不能只比较战斗帧：换进程、下一局以及同帧多次发布都可能重复该帧号。
    return all(getattr(frame, key) == getattr(state, key) for key in (
        "gameProcessId", "sampleSerial", "publisherTickMs", "battleFrame", "currentRound", "initialized"))


class ResourceMemoryClient(SharedMemoryClient):
    state_type = ResourceSharedState
    protocol_magic = 0x53445231
    protocol_version = 1

    def __init__(self, pid):
        self.mapping_name = f"Local\\SokuDataBridge.Resources.v1.{pid}"
        super().__init__()
        if ct.sizeof(ResourceFrame) != 136 or ct.sizeof(ResourceSharedState) != 560:
            raise RuntimeError("技能资源结构对齐不匹配，不能读取游戏内存")

    def read_matching(self, state):
        shared = self._read_copy()
        if shared is None:
            return None
        candidate = shared.frames[int(state.sampleSerial) % 4]
        return candidate if matches(candidate, state) else None


class LiveStateClient:
    """读者只接受同一次采集的战斗状态和资源；历史模型直接使用原 v3 通道。"""

    def __init__(self, require_resources: bool):
        self.state = SharedMemoryClient()
        self.resources = None
        self.pid = None
        self.require_resources = require_resources
        self.wait_reason = "等待 SokuDataBridge 游戏数据"

    def read_state(self):
        return self.state.read()

    def read(self):
        for _ in range(4):
            snapshot = self.read_state()
            if snapshot is None or not snapshot.payload.initialized:
                self.wait_reason = "未收到 DLL 数据：请启动正常游戏并加载 SokuDataBridge"
                return snapshot
            if not self.require_resources:
                return snapshot
            p = snapshot.payload
            if self.resources is None or self.pid != p.gameProcessId:
                if self.resources is not None:
                    self.resources.close()
                self.pid = int(p.gameProcessId)
                self.resources = ResourceMemoryClient(self.pid)
            if not self.resources.connect():
                self.wait_reason = "缺少技能资源通道：资源版模型需要新版 SokuDataBridge.dll，请更新 DLL 后重启正常游戏"
                return None
            frame = self.resources.read_matching(p)
            if frame is not None:
                self.wait_reason = "已连接同帧技能/卡牌资源"
                return SharedSnapshot(snapshot.state, frame)
        self.wait_reason = "正在等待与战斗状态同一采集序号的技能资源，当前不发送按键"
        return None

    def close(self):
        self.state.close()
        if self.resources is not None:
            self.resources.close()
        self.resources, self.pid = None, None
