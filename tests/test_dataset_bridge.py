"""使用真实 BC NPZ 读取器核对 IQL 的输入/标签继承，全部数据在临时目录构造。"""
import copy
import json

import numpy as np

from soku_iql.bc_core.action_space import ACTION_SCHEMA
from soku_iql.bc_core.config import DEFAULTS
from soku_iql.bc_core.dataset import ReplayStore
from soku_iql.bc_core.schema import manifest
from soku_iql.dataset import IQLReplayStore


def write_shard(path):
    n = 12
    state = np.zeros((n, 18), np.float32)
    state[:, 4] = state[:, 11] = 1
    state[:, 16] = [100, 100, 90, 90, 90, 90, 100, 100, 100, 100, 100, 100]
    state[:, 17] = [100, 80, 80, 30, 0, 100, 100, 100, 100, 100, 100, 100]
    valid = np.ones(n, bool)
    valid[[5, 11]] = False
    terminal = np.zeros(n, bool)
    terminal[3] = True
    meta = {**manifest(), "action_shift": 1, "continuous_normalized": False}
    raw = dict(metadata_json=np.asarray(json.dumps(meta)), state_continuous=state,
               state_categorical=np.zeros((n, 5), np.int64), tactical_state=np.zeros((n, 9), np.float32),
               episode_id=np.repeat([0, 1], 6).astype(np.int32), transition_valid=valid, terminated=terminal,
               action_horizontal=np.zeros(n, np.int8), action_vertical=np.zeros(n, np.int8),
               action_buttons=np.zeros((n, 6), np.int8), action_duration=np.arange(n, dtype=np.int32) + 1)
    for side in ("self", "opponent"):
        offsets = np.r_[0, np.cumsum(np.arange(n) % 4)].astype(np.int64)
        raw.update({f"{side}_skill_valid_mask": np.full(n, 15, np.int8),
                    f"{side}_skill_variants": np.ones((n, 4), np.int8),
                    f"{side}_object_offsets": offsets,
                    f"{side}_object_numerical": np.zeros((offsets[-1], 8), np.float32),
                    f"{side}_object_categorical": np.zeros((offsets[-1], 2), np.int64)})
    np.savez(path, **raw)


def test_npz_to_bc_and_iql_current_observations_agree(tmp_path):
    for name in ("train.npz", "validation.npz"):
        write_shard(tmp_path / name)
    config = copy.deepcopy(DEFAULTS)
    config["data"]["directory"] = str(tmp_path)
    config["reward"] = dict(damage_dealt=.001, damage_taken=.001)
    config["training"].update(batch_size=16, sequence_length=3, burn_in=31, replays_per_batch=1)
    split = dict(train=["train.npz"], validation=["validation.npz"],
                 files={"train.npz": "a", "validation.npz": "b"}, sha256="fixed")
    normalization = dict(split_hash="fixed", action_schema=ACTION_SCHEMA, action_shift=1)
    for group, size in (("state", 18), ("objects", 8), ("optional_state", 4)):
        normalization[group] = dict(mean=[0.] * size, std=[1.] * size)
    normalization["optional_state"]["counts"] = [0] * 4
    bc = ReplayStore(config, split, lambda _: None, normalization=normalization)
    iql = IQLReplayStore(config, split, lambda _: None, normalization=normalization)
    old = bc.sample(np.random.default_rng(123), config["training"])
    new = iql.sample(np.random.default_rng(123), config["training"])
    np.testing.assert_array_equal(old["mask"], new["mask"])
    np.testing.assert_array_equal(old["joint_action_id"], new["action"])
    np.testing.assert_array_equal(old["burn_lengths"], new["burn_lengths"])
    for key, value in old["observation"].items():
        np.testing.assert_array_equal(value[:, :31], new["observation"][key][:, :31])
        np.testing.assert_array_equal(value[:, 31:][old["mask"]], new["observation"][key][:, 31:-1][new["mask"]])
    shard = iql.get("train.npz")
    np.testing.assert_allclose(shard["iql_reward"][:6], [.02, -.01, .05, .03, 0, 0], atol=1e-7)
    assert shard["iql_terminal"][3]
    assert iql.normalization == bc.normalization == normalization
