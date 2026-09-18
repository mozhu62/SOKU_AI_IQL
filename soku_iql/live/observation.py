from __future__ import annotations

import numpy as np

from ..bc_core.schema import STATE_CONTINUOUS_FEATURES, STATE_CATEGORICAL_FEATURES
from .resources import build_resources
from .input_history import ControllerHistory
from ..bc_core.resources import normalization_arrays
from ..bc_core.action_space import ACTION_SCHEMA
from ..bc_core.weather import compress_weather


class ObservationBuilder:
    """严格对应 ReplayStore.observation；使用同一训练集归一化及实际控制历史。"""

    def __init__(self, normalization, player_side, positive_down=True):
        self.side = player_side
        self.positive_down = positive_down
        self.resource_summary = None
        self.norm = normalization_arrays(normalization)
        if normalization.get("action_schema") != ACTION_SCHEMA:
            raise ValueError("checkpoint 动作历史归一化 schema 不兼容")
        self.optional_enabled = np.asarray(normalization["optional_state"]["counts"]) > 0
        self.history = ControllerHistory(normalization.get("action_shift"))

    def reset(self):
        self.history.reset()
        self.resource_summary = None

    def observe_inputs(self, payload):
        player = self.players(payload)[0]
        sign = lambda x: (x > 0) - (x < 0)
        direction = 5 + sign(int(player.inputHorizontal)) - 3 * sign(int(player.inputVertical)) * (1 if self.positive_down else -1)
        buttons = [int(getattr(player, key) > 0) for key in
                   ("inputA", "inputD", "inputB", "inputC")]
        self.history.observe((int(payload.gameProcessId), int(payload.currentRound), int(payload.battleFrame)),
                             direction, buttons)

    def players(self, payload):
        return (payload.left, payload.right) if self.side == "left" else (payload.right, payload.left)

    def normalize(self, values, group):
        values = np.asarray(values, dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError("实时有效状态出现 NaN/Inf，已停止发送按键")
        mean, std = self.norm[group]
        return np.clip((values - mean) / std, -10, 10).astype(np.float32)

    def objects(self, owner, player):
        count = min(int(owner.objectCount), len(owner.objects))
        objects = list(owner.objects[:count])
        # 双方对象都按相对当前控制角色的距离选择；浮点精度、并列排序与 NPZ 转换一致。
        dx = np.asarray([x.positionX for x in objects], np.float32) - np.float32(player.positionX)
        dy = np.asarray([x.positionY for x in objects], np.float32) - np.float32(player.positionY)
        order = np.lexsort((np.arange(count), np.hypot(dx, dy)))[:3]
        numeric, category = np.zeros((3, 8), np.float32), np.zeros((3, 2), np.int64)
        mask = np.arange(3) < len(order)
        threat = False
        for slot, index in enumerate(order):
            obj = objects[index]
            raw = [dx[index], dy[index], np.float32(obj.speedX) - np.float32(player.speedX),
                   np.float32(obj.speedY) - np.float32(player.speedY), obj.direction,
                   obj.actionFrameCount, obj.hitstop, obj.hitCount]
            numeric[slot] = self.normalize(raw, "objects")
            category[slot] = [obj.action, obj.actionBlockId]
            threat |= bool(obj.frameDataAvailable) and any(
                box.valid and not rotation.valid for box, rotation in zip(obj.hitBoxes, obj.hitRotationBoxes))
        return numeric, category, mask, threat

    def build(self, payload, resources=None):
        self.observe_inputs(payload)
        self.resource_summary = None
        player, opponent = self.players(payload)
        continuous, categorical = {}, {"active_weather": compress_weather(int(payload.activeWeather))}
        for side, entity in (("self", player), ("opponent", opponent)):
            for name, field in (("position_x", "positionX"), ("position_y", "positionY"),
                                ("speed_x", "speedX"), ("speed_y", "speedY"), ("direction", "direction"),
                                ("current_spirit", "currentSpirit"), ("action_frame_count", "actionFrameCount")):
                continuous[f"{side}_{name}"] = getattr(entity, field)
            continuous[f"{side}_hp"] = max(0, int(entity.hp))
            categorical[f"{side}_action"] = int(entity.action)
            categorical[f"{side}_action_block_id"] = int(entity.actionBlockId)
        continuous["relative_x"] = np.float32(opponent.positionX) - np.float32(player.positionX)
        continuous["relative_y"] = np.float32(opponent.positionY) - np.float32(player.positionY)
        result = {
            "state_continuous": self.normalize([continuous[key] for key in STATE_CONTINUOUS_FEATURES], "state"),
            "state_categorical": np.asarray([categorical[key] for key in STATE_CATEGORICAL_FEATURES], np.int64),
        }
        result.update(self.history.observation())
        optional = np.asarray([player.maxSpirit, player.hitstop, opponent.maxSpirit, opponent.hitstop], np.float32)
        result["state_optional_mask"] = self.optional_enabled.copy()
        result["state_optional_continuous"] = np.where(self.optional_enabled,
                self.normalize(optional, "optional_state"), 0).astype(np.float32)
        threat = False
        for side, entity in (("self", player), ("opponent", opponent)):
            numeric, category, mask, active = self.objects(entity, player)
            result.update({f"{side}_object_numerical": numeric, f"{side}_object_categorical": category,
                           f"{side}_object_mask": mask})
            if side == "opponent":
                threat = active
        flag = lambda entity, bit: bool(entity.frameDataAvailable and entity.frameFlags & (1 << bit))
        result["tactical_state"] = np.asarray([
            flag(player, 11), flag(opponent, 11), flag(player, 10), flag(opponent, 10), threat,
            50 <= player.action < 150, flag(player, 2), 50 <= opponent.action < 150, flag(opponent, 2),
        ], np.float32)
        values, self.resource_summary = build_resources(payload, resources, self.side)
        result.update(values)
        return result
