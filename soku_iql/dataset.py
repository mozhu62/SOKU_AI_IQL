from __future__ import annotations

import numpy as np

from .bc_core.dataset import ReplayStore, read_shard, split_replays
from .bc_core.schema import STATE_CONTINUOUS_FEATURES


def fixed_split(config, progress, cancelled=lambda: False):
    from pathlib import Path
    if not Path(config["data"]["split_file"]).is_file():
        raise FileNotFoundError("必须提供 BC 已使用的固定 split_file，IQL 不重新划分数据")
    return split_replays(config, progress, cancelled)


class IQLReplayStore(ReplayStore):
    def get(self, name):
        with self.lock:
            if name in self.cache:
                self.hits += 1
                self.cache.move_to_end(name)
                return self.cache[name]
            self.misses += 1
        path = self.root / name
        shard = read_shard(path, self.config["data"]["vertical_positive_is_down"])
        with np.load(path, allow_pickle=False) as raw:
            terminal = raw["terminated"].astype(bool)
        state = shard["state_continuous"]
        own = state[:, STATE_CONTINUOUS_FEATURES.index("self_hp")]
        enemy = state[:, STATE_CONTINUOUS_FEATURES.index("opponent_hp")]
        valid = np.zeros(len(state), bool)
        for start, end in shard["segments"]:
            valid[start:end] = True
        reward = np.zeros(len(state), np.float32)
        reward[:-1] = (np.maximum(enemy[:-1] - enemy[1:], 0) * self.config["reward"]["damage_dealt"]
                       - np.maximum(own[:-1] - own[1:], 0) * self.config["reward"]["damage_taken"])
        reward[~valid] = 0
        if not np.isfinite(reward).all():
            raise ValueError(f"伤害奖励含非有限值：{name}")
        # 终局只关闭 bootstrap；片段末尾若存在真实后继且未终局，仍使用该后继的 V。
        shard.update(iql_reward=reward, iql_terminal=terminal)
        size = sum(value.nbytes for value in shard.values())
        with self.lock:
            if name in self.cache:
                return self.cache[name]
            while self.cache and self.bytes + size > self.budget:
                _, old = self.cache.popitem(last=False)
                self.bytes -= sum(value.nbytes for value in old.values())
            if size <= self.budget:
                self.cache[name] = shard
                self.bytes += size
        return shard

    def sample(self, rng, cfg, split="train"):
        names = self.split[split]
        weights = np.asarray([self.info[name]["transitions"] for name in names], np.float64)
        batch, length, burn = cfg["batch_size"], cfg["sequence_length"], cfg["burn_in"]
        chosen = rng.choice(len(names), min(batch, cfg["replays_per_batch"]), p=weights / weights.sum())
        pieces = []
        for shard_id, rows in zip(chosen, np.array_split(np.arange(batch), len(chosen))):
            shard = self.get(names[shard_id])
            picks = rng.integers(0, shard["segment_cumulative"][-1], len(rows))
            segment = np.searchsorted(shard["segment_cumulative"], picks, side="right")
            starts, ends = shard["segments"][segment].T
            prior = np.r_[0, shard["segment_cumulative"][:-1]][segment]
            positions = starts + picks - prior
            burn_lengths = np.minimum(burn, positions - starts)
            prefix = positions[:, None] - burn_lengths[:, None] + np.arange(burn)
            main = positions[:, None] + np.arange(length + 1)
            # 多取真实后继一帧，所有网络共用同一时序窗口；禁止独立 next_state 丢失历史。
            indices = np.concatenate((np.minimum(prefix, positions[:, None]), np.minimum(main, ends[:, None])), 1)
            labels = np.minimum(main[:, :-1], ends[:, None] - 1)
            mask = main[:, :-1] < ends[:, None]
            pieces.append(dict(observation=self.observation(shard, indices), burn_lengths=burn_lengths.astype(np.int64),
                               action=shard["joint_action_id"][labels], reward=shard["iql_reward"][labels],
                               terminal=shard["iql_terminal"][labels], mask=mask,
                               previous_action=shard["previous_expert_action_id"][labels]))
        result = {k: np.concatenate([x[k] for x in pieces]) for k in pieces[0] if k != "observation"}
        result["observation"] = {k: np.concatenate([x["observation"][k] for x in pieces]) for k in pieces[0]["observation"]}
        return result
