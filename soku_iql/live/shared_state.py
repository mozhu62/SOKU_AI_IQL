from __future__ import annotations

import ctypes
import sys
from dataclasses import dataclass
from ctypes import wintypes


SHARED_MEMORY_NAME = "Local\\SokuDataBridge.State.v3"
PROTOCOL_MAGIC = 0x53444D31
PROTOCOL_VERSION = 3
MAX_HAND_CARDS = 16
MAX_DECK_CARDS = 32
MAX_HIT_BOXES = 5
MAX_OBJECTS_PER_PLAYER = 128

FILE_MAP_READ = 0x0004


class BoxState(ctypes.Structure):
    _fields_ = [
        ("left", ctypes.c_int32),
        ("top", ctypes.c_int32),
        ("right", ctypes.c_int32),
        ("bottom", ctypes.c_int32),
        ("valid", ctypes.c_uint32),
    ]


class CardState(ctypes.Structure):
    _fields_ = [
        ("id", ctypes.c_uint32),
        ("cost", ctypes.c_uint32),
    ]


class RotationBoxState(ctypes.Structure):
    _fields_ = [
        ("point1X", ctypes.c_int32),
        ("point1Y", ctypes.c_int32),
        ("point2X", ctypes.c_int32),
        ("point2Y", ctypes.c_int32),
        ("valid", ctypes.c_uint32),
    ]


class ObjectState(ctypes.Structure):
    _fields_ = [
        ("objectAddress", ctypes.c_uint32),
        ("ownerAddress", ctypes.c_uint32),
        ("owner2Address", ctypes.c_uint32),
        ("opponentAddress", ctypes.c_uint32),
        ("positionX", ctypes.c_float),
        ("positionY", ctypes.c_float),
        ("speedX", ctypes.c_float),
        ("speedY", ctypes.c_float),
        ("gravityX", ctypes.c_float),
        ("gravityY", ctypes.c_float),
        ("centerX", ctypes.c_float),
        ("centerY", ctypes.c_float),
        ("direction", ctypes.c_int32),
        ("hp", ctypes.c_int32),
        ("action", ctypes.c_uint32),
        ("actionBlockId", ctypes.c_uint32),
        ("animationCounter", ctypes.c_uint32),
        ("animationSubFrame", ctypes.c_uint32),
        ("actionFrameCount", ctypes.c_uint32),
        ("hitstop", ctypes.c_uint32),
        ("hitCount", ctypes.c_int32),
        ("hurtBoxCount", ctypes.c_uint32),
        ("hitBoxCount", ctypes.c_uint32),
        ("frameDataAvailable", ctypes.c_uint32),
        ("frameNumber", ctypes.c_uint32),
        ("frameFlags", ctypes.c_uint32),
        ("attackFlags", ctypes.c_uint32),
        ("frameDamage", ctypes.c_int32),
        ("frameSpiritDamage", ctypes.c_int32),
        ("collisionBox", BoxState),
        ("hurtBoxes", BoxState * MAX_HIT_BOXES),
        ("hitBoxes", BoxState * MAX_HIT_BOXES),
        ("hurtRotationBoxes", RotationBoxState * MAX_HIT_BOXES),
        ("hitRotationBoxes", RotationBoxState * MAX_HIT_BOXES),
    ]


class PlayerState(ctypes.Structure):
    _fields_ = [
        ("characterId", ctypes.c_int32),
        ("playerIndex", ctypes.c_int32),
        ("direction", ctypes.c_int32),
        ("positionX", ctypes.c_float),
        ("positionY", ctypes.c_float),
        ("speedX", ctypes.c_float),
        ("speedY", ctypes.c_float),
        ("hp", ctypes.c_int32),
        ("currentSpirit", ctypes.c_int32),
        ("maxSpirit", ctypes.c_int32),
        ("action", ctypes.c_uint32),
        ("actionBlockId", ctypes.c_uint32),
        ("animationCounter", ctypes.c_uint32),
        ("animationSubFrame", ctypes.c_uint32),
        ("actionFrameCount", ctypes.c_uint32),
        ("hitstop", ctypes.c_uint32),
        ("guardSucceed", ctypes.c_uint32),
        ("grazeTimer", ctypes.c_uint32),
        ("comboRate", ctypes.c_float),
        ("comboHits", ctypes.c_int32),
        ("comboDamage", ctypes.c_int32),
        ("comboLimit", ctypes.c_int32),
        ("cardGauge", ctypes.c_int32),
        ("cardCount", ctypes.c_int32),
        ("score", ctypes.c_int32),
        ("selectedCard", ctypes.c_int32),
        ("selectedCardData", CardState),
        ("handCapacity", ctypes.c_int32),
        ("handCount", ctypes.c_int32),
        ("handCardsUsed", ctypes.c_int32),
        ("handCards", CardState * MAX_HAND_CARDS),
        ("remainingDeckCount", ctypes.c_int32),
        ("originalDeckCount", ctypes.c_int32),
        ("remainingDeck", ctypes.c_uint32 * MAX_DECK_CARDS),
        ("originalDeck", ctypes.c_uint32 * MAX_DECK_CARDS),
        ("frameDataAvailable", ctypes.c_uint32),
        ("frameNumber", ctypes.c_uint32),
        ("frameFlags", ctypes.c_uint32),
        ("attackFlags", ctypes.c_uint32),
        ("frameDamage", ctypes.c_int32),
        ("frameSpiritDamage", ctypes.c_int32),
        ("collisionBox", BoxState),
        ("hurtBoxes", BoxState * MAX_HIT_BOXES),
        ("hitBoxes", BoxState * MAX_HIT_BOXES),
        ("hurtRotationBoxes", RotationBoxState * MAX_HIT_BOXES),
        ("hitRotationBoxes", RotationBoxState * MAX_HIT_BOXES),
        ("totalObjectCount", ctypes.c_uint32),
        ("objectCount", ctypes.c_uint32),
        ("objects", ObjectState * MAX_OBJECTS_PER_PLAYER),
        ("inputHorizontal", ctypes.c_int32),
        ("inputVertical", ctypes.c_int32),
        ("inputA", ctypes.c_int32),
        ("inputB", ctypes.c_int32),
        ("inputC", ctypes.c_int32),
        ("inputD", ctypes.c_int32),
        ("inputChangeCard", ctypes.c_int32),
        ("inputSpellCard", ctypes.c_int32),
    ]


class StatePayload(ctypes.Structure):
    _fields_ = [
        ("sampleSerial", ctypes.c_uint64),
        ("publisherTickMs", ctypes.c_uint64),
        ("gameProcessId", ctypes.c_uint32),
        ("initialized", ctypes.c_uint32),
        ("inBattle", ctypes.c_uint32),
        ("sceneId", ctypes.c_int32),
        ("battleMode", ctypes.c_int32),
        ("battleSubMode", ctypes.c_int32),
        ("stageId", ctypes.c_int32),
        ("battleFrame", ctypes.c_uint32),
        ("matchState", ctypes.c_int32),
        ("currentRound", ctypes.c_int32),
        ("activeWeather", ctypes.c_int32),
        ("displayedWeather", ctypes.c_int32),
        ("weatherCounter", ctypes.c_uint32),
        ("left", PlayerState),
        ("right", PlayerState),
    ]


class SharedState(ctypes.Structure):
    _fields_ = [
        ("magic", ctypes.c_uint32),
        ("protocolVersion", ctypes.c_uint32),
        ("structureSize", ctypes.c_uint32),
        ("writeSequence", ctypes.c_int32),
        ("payload", StatePayload),
    ]


class SharedMemoryProtocolError(RuntimeError):
    pass


class SharedMemoryNotReady(RuntimeError):
    pass


@dataclass(frozen=True)
class SharedSnapshot:
    state: SharedState
    resources: object | None = None

    @property
    def payload(self) -> StatePayload:
        return self.state.payload

    @property
    def sequence(self) -> int:
        return int(self.state.writeSequence)


class SharedMemoryClient:
    """沿用 DQN 的 v3 协议布局，只读采集 DLL；不依赖旧训练项目。"""

    state_type = SharedState
    mapping_name = SHARED_MEMORY_NAME
    protocol_magic = PROTOCOL_MAGIC
    protocol_version = PROTOCOL_VERSION

    def __init__(self) -> None:
        if sys.platform != "win32":
            raise OSError("实时共享内存客户端只能在 Windows 上运行")
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._configure_api()
        self._mapping: int | None = None
        self._address: int | None = None

    def _configure_api(self) -> None:
        self._kernel32.OpenFileMappingW.argtypes = [
            wintypes.DWORD,
            wintypes.BOOL,
            wintypes.LPCWSTR,
        ]
        self._kernel32.OpenFileMappingW.restype = wintypes.HANDLE
        self._kernel32.MapViewOfFile.argtypes = [
            wintypes.HANDLE,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.DWORD,
            ctypes.c_size_t,
        ]
        self._kernel32.MapViewOfFile.restype = ctypes.c_void_p
        self._kernel32.UnmapViewOfFile.argtypes = [ctypes.c_void_p]
        self._kernel32.UnmapViewOfFile.restype = wintypes.BOOL
        self._kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        self._kernel32.CloseHandle.restype = wintypes.BOOL

    def connect(self) -> bool:
        if self._address is not None:
            return True
        mapping = self._kernel32.OpenFileMappingW(
            FILE_MAP_READ,
            False,
            self.mapping_name,
        )
        if not mapping:
            return False
        address = self._kernel32.MapViewOfFile(
            mapping,
            FILE_MAP_READ,
            0,
            0,
            ctypes.sizeof(self.state_type),
        )
        if not address:
            self._kernel32.CloseHandle(mapping)
            return False
        self._mapping = int(mapping)
        self._address = int(address)
        try:
            self._validate_protocol()
        except SharedMemoryNotReady:
            self.close()
            return False
        except Exception:
            self.close()
            raise
        return True

    def _validate_protocol(self) -> None:
        assert self._address is not None
        header = self.state_type.from_address(self._address)
        expected_size = ctypes.sizeof(self.state_type)
        # 映射可先于 DLL 完成头部初始化被打开，短暂空头应等待而不是结束实战线程。
        if int(header.structureSize) == 0:
            raise SharedMemoryNotReady("共享内存头正在初始化")
        if int(header.magic) != self.protocol_magic:
            raise SharedMemoryProtocolError(
                f"共享内存 Magic 不一致: 0x{int(header.magic):08X}"
            )
        if int(header.protocolVersion) != self.protocol_version:
            raise SharedMemoryProtocolError(
                "共享内存协议版本不一致: "
                f"dll={int(header.protocolVersion)}, python={self.protocol_version}"
            )
        if int(header.structureSize) != expected_size:
            raise SharedMemoryProtocolError(
                "共享内存结构大小不一致: "
                f"dll={int(header.structureSize)}, python={expected_size}"
            )

    def read(self) -> SharedSnapshot | None:
        copied = self._read_copy()
        return SharedSnapshot(copied) if copied is not None else None

    def _read_copy(self):
        if not self.connect():
            return None
        assert self._address is not None
        sequence_address = self._address + self.state_type.writeSequence.offset
        for _ in range(4):
            begin = ctypes.c_int32.from_address(sequence_address).value
            if begin & 1:
                continue
            raw = ctypes.string_at(self._address, ctypes.sizeof(self.state_type))
            end = ctypes.c_int32.from_address(sequence_address).value
            if begin == end and not (end & 1):
                copied = self.state_type.from_buffer_copy(raw)
                if copied.structureSize == 0:
                    return None
                if (copied.magic != self.protocol_magic or copied.protocolVersion != self.protocol_version
                        or copied.structureSize != ctypes.sizeof(self.state_type)):
                    raise SharedMemoryProtocolError("读取期间共享内存协议发生变化，请重连游戏")
                return copied
        return None

    def close(self) -> None:
        if self._address is not None:
            self._kernel32.UnmapViewOfFile(ctypes.c_void_p(self._address))
            self._address = None
        if self._mapping is not None:
            self._kernel32.CloseHandle(wintypes.HANDLE(self._mapping))
            self._mapping = None

    def __enter__(self) -> "SharedMemoryClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
