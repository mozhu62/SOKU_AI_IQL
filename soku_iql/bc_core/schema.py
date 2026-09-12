from __future__ import annotations


CQL_REPLAY_SCHEMA = "soku_cql_raw_axes_action_resources_v4"
POLICY_INPUT_SCHEMA = "soku_bc_joint144_observation_v1"
RESOURCE_INPUT_SCHEMA = "soku_bc_skill_variants_v1"
CQL_TACTICAL_SCHEMA = "soku_cql_replay_tactical_v1"
SKILL_COMMANDS = ("236", "623", "214", "22")
MAX_HAND_CARDS = 16
OPTIONAL_STATE_FEATURES = ("self_max_spirit", "self_hitstop", "opponent_max_spirit", "opponent_hitstop")

STATE_CONTINUOUS_FEATURES = (
    "self_position_x", "self_position_y", "self_speed_x", "self_speed_y",
    "self_direction", "self_current_spirit", "self_action_frame_count",
    "opponent_position_x", "opponent_position_y", "opponent_speed_x", "opponent_speed_y",
    "opponent_direction", "opponent_current_spirit", "opponent_action_frame_count",
    "relative_x", "relative_y", "self_hp", "opponent_hp",
)
STATE_CATEGORICAL_FEATURES = (
    "self_action", "self_action_block_id", "opponent_action",
    "opponent_action_block_id", "active_weather",
)
TACTICAL_FEATURES = (
    "self_guarding", "opponent_guarding", "self_graze_active", "opponent_graze_active",
    "opponent_projectile_attack_active", "self_hurt_state", "self_airborne_flag",
    "opponent_hurt_state", "opponent_airborne_flag",
)
OBJECT_NUMERICAL_FEATURES = (
    "relative_position_x", "relative_position_y", "relative_speed_x", "relative_speed_y",
    "direction", "action_frame_count", "hitstop", "hit_count",
)
OBJECT_CATEGORICAL_FEATURES = ("action", "action_block_id")
ACTION_DIRECTION_FEATURES = ("horizontal", "vertical", "duration")
BUTTON_FEATURES = ("melee", "dash", "light_projectile", "heavy_projectile")
# v4 是既有文件格式，不随本次模型动作实验重新定义。
RAW_BUTTON_FEATURES = (*BUTTON_FEATURES, "change_card", "use_spell_card")

GLOBAL_COLUMNS = (
    "sample_serial", "battle_frame", "initialized", "in_battle",
    "battle_sub_mode", "match_state", "current_round", "active_weather",
)
PLAYER_COLUMNS = (
    "character_id", "direction", "position_x", "position_y", "speed_x", "speed_y",
    "hp", "current_spirit", "action", "action_block_id", "action_frame_count",
    "input_horizontal", "input_vertical", "input_a", "input_b", "input_c", "input_d",
    "input_change_card", "input_spell_card", "frame_data_available", "frame_flags",
    "card_gauge", "card_count", "selected_card_index", "selected_card_id", "selected_card_cost",
    "hand_capacity", "hand_count", "hand_cards_used", "skill_valid_mask",
    *(f"skill_{command}_{field}" for command in SKILL_COMMANDS
      for field in ("variant", "level", "effective_level")),
    *(f"hand_{index}_{field}" for index in range(MAX_HAND_CARDS) for field in ("id", "cost")),
)
OBJECT_COLUMNS = (
    "sample_serial", "battle_frame", "side", "object_index", "position_x", "position_y",
    "speed_x", "speed_y", "direction", "action", "action_block_id", "action_frame_count",
    "hitstop", "hit_count", "frame_data_available",
    *(f"hit_box_{index}_valid" for index in range(5)),
    *(f"hit_rotation_box_{index}_valid" for index in range(5)),
)


def required_main_columns() -> tuple[str, ...]:
    return GLOBAL_COLUMNS + tuple(
        f"{side}_{field}" for side in ("left", "right") for field in PLAYER_COLUMNS
    )


def manifest() -> dict:
    return {
        "dataset_schema": CQL_REPLAY_SCHEMA,
        "cql_tactical_schema": CQL_TACTICAL_SCHEMA,
        "state_continuous": list(STATE_CONTINUOUS_FEATURES),
        "state_categorical": list(STATE_CATEGORICAL_FEATURES),
        "tactical_state": list(TACTICAL_FEATURES),
        "object_numerical": list(OBJECT_NUMERICAL_FEATURES),
        "object_categorical": list(OBJECT_CATEGORICAL_FEATURES),
        "action_axes": list(ACTION_DIRECTION_FEATURES),
        "button_features": list(RAW_BUTTON_FEATURES),
        "button_storage": "six_independent_uint8_columns",
        "replay_resources": {
            "skill_commands": list(SKILL_COMMANDS),
            "skill_fields": ["variant", "level", "effective_level"],
            "card_state": ["gauge", "count", "selected_index", "selected_id", "selected_cost",
                           "hand_capacity", "hand_count", "hand_cards_used"],
            "hand_slots": MAX_HAND_CARDS,
        },
    }


def policy_input_manifest() -> dict:
    """原始 v4 NPZ 可复用，但训练输入和动作语义采用独立的新版本，旧权重不能加载。"""
    from .action_space import (
        ACTION_SCHEMA, ACTION_COUNT, START_ACTION_ID, COMBAT_BUTTONS, PREVIOUS_ACTION_VOCAB, RAW_CARD_PROJECTION,
    )
    result = {
        "observation_schema": POLICY_INPUT_SCHEMA,
        "raw_dataset_schema": CQL_REPLAY_SCHEMA,
        "cql_tactical_schema": CQL_TACTICAL_SCHEMA,
        "state_continuous": list(STATE_CONTINUOUS_FEATURES),
        "state_categorical": list(STATE_CATEGORICAL_FEATURES),
        "tactical_state": list(TACTICAL_FEATURES),
        "object_numerical": list(OBJECT_NUMERICAL_FEATURES),
        "object_categorical": list(OBJECT_CATEGORICAL_FEATURES),
        "optional_state_continuous": list(OPTIONAL_STATE_FEATURES),
        "optional_state_mask": "原始字段存在且训练集有记录时才有效；缺失不是数值零",
        "objects_per_side": 3,
        "previous_joint_action_id": {"vocabulary": PREVIOUS_ACTION_VOCAB, "start_pad": START_ACTION_ID, "offset": -1},
        "previous_action_duration": "clip(direction_axes_duration[t-1], 0, 60)/60; boundary=0",
        "action_schema": ACTION_SCHEMA,
        "joint_action_count": ACTION_COUNT,
        "combat_bit_order": list(COMBAT_BUTTONS),
        "raw_card_projection": RAW_CARD_PROJECTION,
        "joint_encoding": "(direction-1)*16+combat_mask",
        "continuous_normalization": "training_split_only; previous duration uses fixed 60-frame scaling",
    }
    result["resource_inputs"] = {
            "schema": RESOURCE_INPUT_SCHEMA,
            "sides": ["self", "opponent"],
            "skill_slots": [f"skill_slot_{i}" for i in range(1, 5)],
            "skill_categorical": ["variant"],
            "skill_tokens": "0=unknown; known raw value + 1",
            "skill_mask": "one boolean per command slot",
            "removed_inputs": ["skill_levels", "skill_effective_levels", "all_card_state"],
        }
    return result

