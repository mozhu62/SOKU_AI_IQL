from __future__ import annotations

import numpy as np

from ..bc_core.action_space import START_ACTION_ID, from_controller


class ControllerHistory:
    """只读取游戏已消费的输入，不记录模型计划发送但可能未生效的按键。"""

    def __init__(self, action_shift):
        if action_shift not in (0, 1):
            raise ValueError("实战缺少与 REP 匹配的 action_shift")
        self.action_shift = action_shift
        self.reset()

    def reset(self):
        self.key = None
        self.direction = None
        self.observed_action = START_ACTION_ID
        self.direction_duration = 0
        self.previous_id = START_ACTION_ID
        self.previous_duration = 0.0

    def observe(self, key, direction, buttons):
        action = from_controller(direction, np.asarray(buttons, np.int64), "游戏实际输入回读")
        if key == self.key:
            return
        continuous = self.key is not None and key[:2] == self.key[:2] and key[2] == self.key[2] + 1
        old_action, old_duration = self.observed_action, self.direction_duration
        self.direction_duration = old_duration + 1 if continuous and direction == self.direction else 1
        if not continuous:
            self.previous_id, self.previous_duration = START_ACTION_ID, 0.0
        elif self.action_shift == 1:
            # 默认 REP 标签 a_t 来自下一采集帧；当前回读输入正是已执行的 a_(t-1)。
            self.previous_id = action
            self.previous_duration = min(self.direction_duration, 60) / 60.0
        else:
            self.previous_id = old_action
            self.previous_duration = min(old_duration, 60) / 60.0
        self.key, self.direction, self.observed_action = key, direction, action

    def observation(self):
        return {"previous_joint_action_id": np.asarray(self.previous_id, np.int64),
                "previous_action_duration": np.asarray([self.previous_duration], np.float32)}
