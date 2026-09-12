from __future__ import annotations

import numpy as np

from ..bc_core.resources import player_resource_observation, validate_player_resources
from ..bc_core.schema import SKILL_COMMANDS
from .resource_state import matches


def build_resources(payload, frame, player_side):
    if frame is None or not matches(frame, payload):
        raise ValueError("模型需要同一采集帧的技能类型，不能用缺省技能补齐推理输入")
    sides = ("left", "right") if player_side == "left" else ("right", "left")
    observation, summary = {}, {"sample_serial": int(payload.sampleSerial), "battle_frame": int(payload.battleFrame)}
    for side, source in zip(("self", "opponent"), sides):
        skills = getattr(frame.resources, source)
        # 保持 DLL 协议结构不变，只读取模型实际使用的 variant 和有效位。
        raw = {
            "skill_valid_mask": np.asarray(skills.validMask, np.int64),
            "skill_variants": np.asarray([s.variant for s in skills.skills], np.int64),
        }
        validate_player_resources(raw, (), side)
        observation.update({f"{side}_{key}": value for key, value in player_resource_observation(raw).items()})
        summary[side] = {
            "skills": [{"slot": index + 1, "command": command,
                        "valid": bool(skills.validMask & (1 << index)), "variant": int(s.variant)}
                       for index, (command, s) in enumerate(zip(SKILL_COMMANDS, skills.skills))]
        }
    return observation, summary
