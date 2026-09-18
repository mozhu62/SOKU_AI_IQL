from __future__ import annotations

import numpy as np


RAW_WEATHER_COUNT = 22
WEATHER_CATEGORY_COUNT = 9
WEATHER_CATEGORY_NAMES = (
    "normal", "花昙", "太阳雨", "风雨", "川雾", "台风", "黄砂", "烈日", "梅雨",
)

# 只让会改变核心对战规则的天气占用独立 token，其余天气统一归入 normal。
SPECIAL_WEATHER_CATEGORIES = {
    5: 1,   # 花昙：打击/擦弹规则
    8: 2,   # 太阳雨（SokuLib: SUN_SHOWER）：防御规则
    10: 3,  # 风雨：移动规则
    12: 4,  # 川雾：距离动力学
    13: 5,  # 台风：受击/防御系统
    16: 6,  # 黄砂：CH/追击规则
    17: 7,  # 烈日：空中风险收益
    18: 8,  # 梅雨：弹墙/弹地规则
}

_WEATHER_LOOKUP = np.zeros(RAW_WEATHER_COUNT, dtype=np.int64)
for _raw_id, _category in SPECIAL_WEATHER_CATEGORIES.items():
    _WEATHER_LOOKUP[_raw_id] = _category


def compress_weather(values):
    """把游戏原始天气 0..21 压缩为 normal+八种特殊天气的 0..8 token。"""
    raw = np.asarray(values)
    if not np.issubdtype(raw.dtype, np.integer) or np.any(raw < 0) or np.any(raw >= RAW_WEATHER_COUNT):
        raise ValueError("active_weather 必须是 SokuLib 的 0..21 天气 ID")
    compressed = _WEATHER_LOOKUP[raw]
    return int(compressed) if compressed.ndim == 0 else compressed


def weather_input_manifest():
    return {
        "vocabulary_size": WEATHER_CATEGORY_COUNT,
        "categories": list(WEATHER_CATEGORY_NAMES),
        "special_raw_id_to_token": {str(key): value for key, value in SPECIAL_WEATHER_CATEGORIES.items()},
        "other_raw_ids": "normal=0",
    }
