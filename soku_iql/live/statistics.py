from __future__ import annotations

import copy
import json
import time
from collections import Counter
from datetime import datetime
from uuid import uuid4

import numpy as np

from ..config import resolve
from ..bc_core.storage import atomic_json
from ..bc_core.action_space import ACTION_SCHEMA, NEUTRAL_ACTION_ID, from_controller, action_catalog


def frame_key(payload):
    return int(payload.gameProcessId), int(payload.currentRound), int(payload.battleFrame)


def readback(player, positive_down):
    sign = lambda x: (x > 0) - (x < 0)
    horizontal = sign(int(player.inputHorizontal))
    vertical = sign(int(player.inputVertical)) * (1 if positive_down else -1)
    return 5 + horizontal - 3 * vertical, tuple(int(getattr(player, key) > 0) for key in (
        "inputA", "inputD", "inputB", "inputC"))


class EvaluationStatistics:
    def __init__(self, config, model):
        self.side = config["environment"]["player_side"]
        self.positive_down = model.vertical_positive_is_down
        self.target = config["rounds"]
        self.active_states = set(config["environment"]["active_match_states"])
        name = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8]
        self.directory = resolve(config["output"]) / name
        self.directory.mkdir(parents=True, exist_ok=False)
        self.metadata = {"schema": "iql_live_evaluation_joint144_v1", "algorithm": model.trained_algorithm, "action_schema": ACTION_SCHEMA,
                         "action_catalog": action_catalog(), "config": copy.deepcopy(config),
                         "checkpoint": model.path, "checkpoint_sha256": model.sha256, "step": model.step,
                         "network_version": model.model.spec["network_version"],
                         "temporal": copy.deepcopy(model.model.spec["temporal"]),
                         "temporal_capture_policy": f"liveframes_v1_real{model.tcn_window.size}_no_padding",
                         "model_inputs": copy.deepcopy(model.model.spec["inputs"]),
                         "device": str(model.device), "action_selection": model.action_selection,
                         "output_semantics": "categorical_logits",
                         "vertical_positive_is_down": self.positive_down,
                         "damage_metric": "同局相邻有效观测的 HP 净扣减，非伤害因果追踪",
                         "started": datetime.now().isoformat()}
        atomic_json(self.directory / "session.json", self.metadata)
        self.rows = []
        self.current = None
        self.previous = None
        self.last_terminal = None
        self.last_action = None
        self.inference_ms = []
        self.session_actions = Counter()
        self.sent = 0
        self.completed = 0
        self.observed_modes = set()

    def players(self, payload):
        return (payload.left, payload.right) if self.side == "left" else (payload.right, payload.left)

    def mark_partial(self, reason):
        if self.current is not None:
            self.current["complete"] = False
            if reason not in self.current["notes"]:
                self.current["notes"].append(reason)

    def cut(self, reason):
        if self.current is not None:
            self.mark_partial(reason)
            self.finish("interrupted", reason)
        self.previous = None
        self.last_action = None

    def finish(self, result, reason):
        if self.current is None:
            return
        row = self.current
        row.update(result=result, outcome_source=reason, damage_difference=row["damage_dealt"] - row["damage_taken"],
                   wall_seconds=round(time.monotonic() - row.pop("started_clock"), 3))
        row["eligible"] = row["complete"] and result in ("win", "loss", "draw")
        self.completed += int(row["eligible"])
        for key in ("joint_counts", "readback_joint_counts", "actual_actions"):
            row[key] = dict(row[key])
        self.rows.append(row)
        self.current = None
        with (self.directory / "rounds.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        self.save()

    def observe(self, payload, controlled, active):
        if controlled and active:
            self.observed_modes.add((int(payload.battleMode), int(payload.battleSubMode)))
        old = self.previous
        player, opponent = self.players(payload)
        changed = old is not None and (frame_key(payload)[:2] != frame_key(old)[:2]
                                      or payload.battleFrame < old.battleFrame)
        outcome, source = None, ""
        if old is not None and payload.inBattle and old.inBattle and payload.gameProcessId == old.gameProcessId:
            own_old, other_old = self.players(old)
            own_score, other_score = player.score > own_old.score, opponent.score > other_old.score
            if own_score or other_score:
                outcome = "draw" if own_score and other_score else "win" if own_score else "loss"
                source = "score_increase"
        if payload.inBattle and (player.hp <= 0 or opponent.hp <= 0):
            outcome = "draw" if player.hp <= 0 and opponent.hp <= 0 else "win" if opponent.hp <= 0 else "loss"
            source = "hp_zero"
        if self.current is not None:
            if not controlled:
                self.cut("控制暂停或未在前台")
            elif changed or not payload.inBattle:
                # 跨局 HP 重置不可当成伤害；未看到终局的局不参与正式比较。
                self.mark_partial("切换边界前未采到终局")
                self.finish(outcome or "interrupted", source or "scene_or_round_changed")
            elif old is not None and payload.battleFrame >= old.battleFrame:
                own_old, other_old = self.players(old)
                self.current["damage_dealt"] += max(0, max(0, int(other_old.hp)) - max(0, int(opponent.hp)))
                self.current["damage_taken"] += max(0, max(0, int(own_old.hp)) - max(0, int(player.hp)))
                gap = int(payload.battleFrame) - int(old.battleFrame)
                self.current["missing_frames"] += max(0, gap - 1)
                self.current["end_frame"] = int(payload.battleFrame)
                if outcome:
                    self.last_terminal = frame_key(payload)[:2]
                    self.finish(outcome, source)
        can_open = controlled and active and player.hp > 0 and opponent.hp > 0
        if self.current is None and can_open and (frame_key(payload)[:2] != self.last_terminal or changed):
            # 只有观察到开局边界且从开局接管，才有完整小局；首次中途接管只作片段。
            clean = old is not None and (changed or not old.inBattle or old.matchState not in self.active_states)
            self.current = {"index": len(self.rows) + 1, "complete": clean,
                            "notes": [] if clean else ["中途接管，未观测完整开局"],
                            "start_frame": int(payload.battleFrame), "end_frame": int(payload.battleFrame),
                            "round": int(payload.currentRound), "damage_dealt": 0, "damage_taken": 0,
                            "decisions": 0, "observed_frames": 0, "missing_frames": 0,
                            "joint_counts": Counter(), "readback_joint_counts": Counter(),
                            "actual_actions": Counter(), "started_clock": time.monotonic()}
            self.last_action = None
        if self.current is not None and active:
            direction, buttons = readback(player, self.positive_down)
            self.current["observed_frames"] += 1
            actual_controller = from_controller(direction, np.asarray(buttons, np.int64), "实战统计回读")
            self.current["readback_joint_counts"][actual_controller] += 1
            action = int(player.action), int(player.actionFrameCount)
            if self.last_action is None or action[0] != self.last_action[0] or action[1] < self.last_action[1]:
                self.current["actual_actions"][action[0]] += 1
            self.last_action = action
        self.previous = payload

    def decision(self, prediction):
        self.sent += 1
        self.session_actions[prediction["joint_action_id"]] += 1
        self.inference_ms.append(prediction["inference_ms"])
        if len(self.inference_ms) > 5000:
            del self.inference_ms[:1000]
        if self.current is not None:
            self.current["decisions"] += 1
            self.current["joint_counts"][prediction["joint_action_id"]] += 1

    def summary(self):
        eligible = [row for row in self.rows if row["eligible"]]
        differences = [row["damage_difference"] for row in eligible]
        return {"completed": len(eligible), "target": self.target, "fragments": len(self.rows) - len(eligible),
                "observed_modes": [list(pair) for pair in sorted(self.observed_modes)],
                "wins": sum(row["result"] == "win" for row in eligible),
                "losses": sum(row["result"] == "loss" for row in eligible),
                "draws": sum(row["result"] == "draw" for row in eligible),
                "win_rate": sum(row["result"] == "win" for row in eligible) / len(eligible) if eligible else None,
                "mean_damage_difference": float(np.mean(differences)) if differences else None,
                "std_damage_difference": float(np.std(differences)) if differences else None,
                "inference_p50_ms": float(np.percentile(self.inference_ms, 50)) if self.inference_ms else None,
                "inference_p95_ms": float(np.percentile(self.inference_ms, 95)) if self.inference_ms else None,
                "decisions": self.sent, "joint_counts": dict(self.session_actions),
                "neutral_fraction": self.session_actions[NEUTRAL_ACTION_ID] / self.sent if self.sent else None,
                "report_directory": str(self.directory)}

    def save(self):
        atomic_json(self.directory / "summary.json", self.summary())

    def snapshot(self):
        return {"summary": self.summary(), "current_round": copy.deepcopy(self.current),
                "rounds": copy.deepcopy(self.rows[-100:])}
