from __future__ import annotations

import numpy as np

from .bc_core.dataset import ReplayStore, read_shard, split_replays
from .bc_core.schema import STATE_CONTINUOUS_FEATURES


def fixed_split(config, progress, cancelled=lambda: False):
    from pathlib import Path
    if not Path(config["data"]["split_file"]).is_file():
        raise FileNotFoundError("必须提供当前 IQL 数据集的固定 split_file，训练启动时不随机重划分")
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
        damage_reward = reward.copy()
        outcome = ko_outcomes(own, enemy, valid, terminal)
        reward += np.where(outcome > 0, self.config['reward'].get('win', 0.0),
                           np.where(outcome < 0, -self.config['reward'].get('loss', 0.0), 0)).astype(np.float32)
        if not np.isfinite(reward).all():
            raise ValueError(f"伤害奖励含非有限值：{name}")
        # 终局只关闭 bootstrap；片段末尾若存在真实后继且未终局，仍使用该后继的 V。
        shard.update(iql_reward=reward, iql_terminal=terminal,
                     diagnostic_damage_reward=damage_reward,
                     diagnostic_win_loss_reward=np.where(outcome > 0, self.config['reward'].get('win', 0.0),
                         np.where(outcome < 0, -self.config['reward'].get('loss', 0.0), 0)).astype(np.float32))
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

    def sample(self, rng, cfg, split="train", diagnostic_metadata=False):
        names = self.split[split]
        weights = np.asarray([self.info[name]["transitions"] for name in names], np.float64)
        batch, length, burn = cfg["batch_size"], cfg["sequence_length"], cfg["burn_in"]
        chosen = rng.choice(len(names), min(batch, cfg["replays_per_batch"]), p=weights / weights.sum())
        pieces = []
        identities = []
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
            # 只读取一个额外真实后继；尾部无效监督仍由 mask 排除。
            indices = np.concatenate((np.minimum(prefix, positions[:, None]), np.minimum(main, ends[:, None])), 1)
            labels = np.minimum(main[:, :length], ends[:, None] - 1)
            mask = main[:, :length] < ends[:, None]
            if diagnostic_metadata:
                identities.extend([{"shard": names[shard_id], "frame": int(labels[row, col]),
                                    "state": shard["state_continuous"][labels[row, col]].tolist(),
                                    "tactical": shard["tactical_state"][labels[row, col]].tolist()}
                                   for row, col in np.argwhere(mask)])
            pieces.append(dict(observation=self.observation(shard, indices), burn_lengths=burn_lengths.astype(np.int64),
                               action=shard["joint_action_id"][labels], reward=shard["iql_reward"][labels],
                               damage_reward=shard["diagnostic_damage_reward"][labels],
                               win_loss_reward=shard["diagnostic_win_loss_reward"][labels],
                               terminal=shard["iql_terminal"][labels], mask=mask,
                               previous_action=shard["previous_expert_action_id"][labels]))
        result = {k: np.concatenate([x[k] for x in pieces]) for k in pieces[0] if k != "observation"}
        result["observation"] = {k: np.concatenate([x["observation"][k] for x in pieces]) for k in pieces[0]["observation"]}
        if diagnostic_metadata:
            result["_diagnostic_meta"] = identities
        return result


def ko_outcomes(own, enemy, valid, terminal):
    """只识别有效终局转移中的首次 KO；不将双倒、缺帧或截断推断为胜负。"""
    outcome = np.zeros(len(own), np.int8)
    eligible = (valid[:-1] & terminal[:-1] & (own[:-1] > 0) & (enemy[:-1] > 0))
    outcome[:-1] = np.where(eligible & (enemy[1:] <= 0) & (own[1:] > 0), 1,
                           np.where(eligible & (own[1:] <= 0) & (enemy[1:] > 0), -1, 0))
    return outcome


def n_step_returns(rewards, terminal, positions, ends, n_step, gamma):
    total = np.zeros(positions.shape, np.float32)
    discount = np.ones(positions.shape, np.float32)
    done = np.zeros(positions.shape, bool)
    steps = np.zeros(positions.shape, np.int64)
    for offset in range(n_step):
        index = positions + offset
        active = (index < ends) & ~done
        safe = np.minimum(index, len(rewards)-1)
        total += np.where(active, discount * rewards[safe], 0)
        discount = np.where(active, discount * gamma, discount)
        steps += active
        done |= active & terminal[safe]
    # ends 是连续有效转移的开区间终点；截断可从最后真实后继 bootstrap，终局不可。
    return total, discount, done, steps
