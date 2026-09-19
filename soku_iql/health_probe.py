"""固定验证探针：独立 RNG、持久化身份及覆盖统计，不改变训练采样。"""
import hashlib
import json
import logging

import numpy as np

from . import health
from .health_config import class_map
from .bc_core.schema import STATE_CONTINUOUS_FEATURES, TACTICAL_FEATURES
from .bc_core.storage import atomic_json

LOGGER = logging.getLogger(__name__)


def fingerprint(batch):
    digest = hashlib.sha256()
    def visit(value, path):
        if isinstance(value, dict):
            for key in sorted(value):
                if key != "_diagnostic_meta":
                    visit(value[key], path + "/" + key)
        else:
            array = np.ascontiguousarray(value)
            digest.update(f"{path}:{array.dtype}:{array.shape}".encode())
            digest.update(array.tobytes())
    visit(batch, "")
    return digest.hexdigest()


def tags(identity, action, cfg):
    state = dict(zip(STATE_CONTINUOUS_FEATURES, identity["state"]))
    tactical = dict(zip(TACTICAL_FEATURES, identity["tactical"]))
    distance = abs(state["relative_x"])
    result = ["near" if distance <= cfg["near_distance"] else "far" if distance >= cfg["far_distance"] else "mid",
              "hp_advantage" if state["self_hp"] > state["opponent_hp"] else
              "hp_disadvantage" if state["self_hp"] < state["opponent_hp"] else "hp_equal",
              "airborne" if tactical["self_airborne_flag"] else "ground",
              "action:" + class_map(cfg)[action]]
    # 只记录数据中明确提供的状态，不用距离或按键臆测压制关系。
    for key in ("self_guarding", "opponent_guarding", "self_hurt_state", "opponent_hurt_state"):
        if tactical[key]:
            result.append(key)
    if identity.get("pressure_event"):
        result.append("pressure_event")
    return result


class FixedProbe:
    def __init__(self, store, config, output):
        self.store, self.config, self.output = store, config, output
        cfg = config["diagnostics"]
        self.sample_config = dict(config["training"], batch_size=cfg["probe_batch_size"],
                                  sequence_length=cfg["probe_sequence_length"],
                                  replays_per_batch=cfg["probe_replays_per_batch"])
        candidates, hashes = [], []
        for index in range(cfg["probe_batches"]):
            batch = self.batch(index)
            hashes.append(fingerprint(batch))
            actions = batch["action"][batch["mask"]]
            for offset, identity in enumerate(batch["_diagnostic_meta"]):
                candidates.append(dict(batch=index, offset=offset, **identity,
                                       action=int(actions[offset]), tags=tags(identity, int(actions[offset]), cfg)))
        groups = {}
        for index, item in enumerate(candidates):
            for tag in item["tags"]:
                groups.setdefault(tag, []).append(index)
        selected, used_ids = [], set()
        # 固定候选池内按覆盖标签轮流选择；未覆盖标签明确记录，不宣称全数据分层采样。
        cursor = 0
        while len(selected) < cfg["probe_max_samples"]:
            progressed = False
            for tag in sorted(groups):
                if cursor >= len(groups[tag]):
                    continue
                progressed = True
                index = groups[tag][cursor]
                item = candidates[index]
                identity = (item["shard"], item["frame"])
                if identity not in used_ids:
                    selected.append(index)
                    used_ids.add(identity)
                if len(selected) >= cfg["probe_max_samples"]:
                    break
            cursor += 1
            if not progressed:
                break
        self.selected = [candidates[index] for index in sorted(selected)]
        self.hashes = hashes
        coverage = {tag: sum(tag in item["tags"] for item in self.selected) for tag in groups}
        expected = ["near", "mid", "far", "hp_advantage", "hp_disadvantage", "airborne", "ground",
                    "self_guarding", "opponent_guarding", "self_hurt_state", "opponent_hurt_state",
                    "pressure_event",
                    *["action:" + name for name in sorted(set(class_map(cfg)))]]
        manifest = dict(version=1, split_hash=store.split["sha256"], seed=cfg["probe_seed"],
                        sampling={key: self.sample_config[key] for key in
                                  ("batch_size", "sequence_length", "burn_in", "replays_per_batch")},
                        batch_hashes=hashes, selected=self.selected, coverage=coverage,
                        missing_coverage=[tag for tag in expected if not coverage.get(tag)],
                        unsupported_tags=["flight_action", "defense_action", "skill_action", "spell_action"],
                        class_map=class_map(cfg), near_distance=cfg["near_distance"], far_distance=cfg["far_distance"])
        encoded = json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode()
        self.id = hashlib.sha256(encoded).hexdigest()
        manifest["probe_id"] = self.id
        path = output / f"probe_manifest_{self.id[:16]}.json"
        if path.exists():
            if json.loads(path.read_text(encoding="utf-8")) != manifest:
                raise ValueError("固定 probe 清单发生变化")
        else:
            atomic_json(path, manifest)
        LOGGER.info("固定 probe=%s，样本=%d，缺失覆盖=%s", self.id, len(self.selected), manifest["missing_coverage"])
        self.last_step = None
        self.previous = None
        history = output / "health_probe.jsonl"
        if history.exists():
            with history.open(encoding="utf-8") as stream:
                for line in stream:
                    try:
                        row = json.loads(line)
                    except ValueError:
                        continue
                    if row.get("probe_id") == self.id:
                        self.previous = row

    def batch(self, index):
        rng = np.random.default_rng(np.random.SeedSequence([self.config["diagnostics"]["probe_seed"], index, 715]))
        return self.store.sample(rng, self.sample_config, "validation", diagnostic_metadata=True)

    def evaluate(self, learner, step):
        if self.last_step == step:
            return
        packets = []
        for index in range(len(self.hashes)):
            selected = [item for item in self.selected if item["batch"] == index]
            if not selected:
                continue
            batch = self.batch(index)
            if fingerprint(batch) != self.hashes[index]:
                raise ValueError("Probe 内容不一致，停止该次诊断，禁止比较不同状态")
            packet = learner.validate_batch(batch, health_diagnostics=True)["_health"]
            offsets = [item["offset"] for item in selected]
            packets.append({key: value[offsets] for key, value in packet.items()})
            for item in selected:
                offset = item["offset"]
                point = {key: value[offset] for key, value in packet.items()}
                logits = point["logits"].astype(np.float64)
                probability = np.exp(logits - logits.max())
                point.update(probabilities=probability / probability.sum(), step=step, probe_id=self.id,
                             sample_id=f"{item['shard']}:{item['frame']}", tags=item["tags"],
                             expert_history_conditioned=True)
                health.append(self.output / "probe_samples.jsonl", point)
        row = health.summarize(health.merge(packets), self.config, "fixed_probe_candidate_weights")
        row.update(step=step, probe_id=self.id,
                   sampling_note="固定候选池覆盖抽样；并非自然总体频率，验证使用全部有效样本，不执行Actor随机筛选")
        if self.previous and self.previous.get("step", step) < step:
            row["delta_from_previous_probe"] = {key: row[key] - self.previous[key] for key in
                ("predicted_attack_ratio", "predicted_switch_rate", "attack_effective_weight_ratio", "adv_std", "weight_clip_ratio")
                if row.get(key) is not None and self.previous.get(key) is not None}
            row["previous_probe_step"] = self.previous["step"]
        health.append(self.output / "health_probe.jsonl", row)
        self.previous, self.last_step = row, step
        return row
