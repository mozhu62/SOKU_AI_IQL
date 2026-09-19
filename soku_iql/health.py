"""只读诊断：复用实际更新张量，按有效样本合并后计算分位数。"""
import json
import logging
import math
from pathlib import Path

import numpy as np
import torch

from .health_config import class_map
from .bc_core.action_space import NEUTRAL_ACTION_ID

LOGGER = logging.getLogger(__name__)


@torch.no_grad()
def capture(batch, q1, q2, target_q1, target_q2, value, target, advantage,
            weights, keyframe_weights, changed, eligible, actor_mask, logits,
            neutral_weight=1.0, actor_updated=False):
    """CPU 快照不保留计算图；训练取更新时的权重，不用更新后重算值代替。"""
    mask, actions = batch["mask"].bool(), batch["action"].long()
    prediction = logits.argmax(-1) if logits is not None else torch.full_like(actions, -1)
    pair = torch.zeros_like(mask)
    if mask.shape[1] > 1:
        pair[:, 1:] = (mask[:, 1:] & mask[:, :-1] & ~batch["terminal"][:, :-1]
                       & (batch["previous_action"][:, 1:] == actions[:, :-1]))
    expert_switch = torch.zeros_like(mask)
    predicted_switch = torch.zeros_like(mask)
    expert_switch[:, 1:] = actions[:, 1:] != actions[:, :-1]
    predicted_switch[:, 1:] = prediction[:, 1:] != prediction[:, :-1]
    category = torch.where(actions == NEUTRAL_ACTION_ID, neutral_weight, 1.0)
    denominator = (keyframe_weights * category * actor_mask).sum()
    values = dict(action=actions, prediction=prediction, q1=q1, q2=q2,
                  target_q1=target_q1, target_q2=target_q2, v=value,
                  td_target=target, reward=batch["reward"], adv=advantage,
                  weight=weights, keyframe_weight=keyframe_weights,
                  final_weight=keyframe_weights * weights,
                  effective_weight=keyframe_weights * weights * category * actor_mask,
                  denominator_weight=keyframe_weights * category * actor_mask,
                  actor_selected=actor_mask, changed=changed, history_eligible=eligible,
                  switch_pair=pair & (logits is not None), expert_switch=expert_switch,
                  predicted_switch=predicted_switch,
                  actor_updated=torch.full_like(mask, actor_updated))
    values["loss_coefficient"] = values["effective_weight"] / denominator.clamp_min(1e-30)
    values["applied_loss_coefficient"] = values["loss_coefficient"] * actor_updated
    for key in ("damage_reward", "wrong_block_reward", "pressure_event",
                "pressure_spirit_loss", "pressure_reward", "far_distance_event",
                "horizontal_distance", "far_distance_penalty", "win_loss_reward"):
        if key in batch:
            values[key] = batch[key]
    for key in ("n_step_reward", "n_step_discount", "n_step_actual"):
        if key in batch:
            values[key] = batch[key]
    result = {key: value.detach()[mask].cpu().numpy() for key, value in values.items()}
    if logits is not None:
        result["logits"] = logits.detach()[mask].cpu().numpy()
    return result


def merge(packets):
    keys = set.intersection(*(set(packet) for packet in packets))
    return {key: np.concatenate([packet[key] for packet in packets]) for key in keys}


def distribution(values, prefix, percentiles=(5, 50, 95)):
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    result = {f"{prefix}_{key}": None for key in ("mean", "std", "min", "max")}
    result.update({f"{prefix}_p{p:02d}": None for p in percentiles})
    result[f"{prefix}_nonfinite_count"] = int(len(values) - len(finite))
    if len(finite):
        result.update({f"{prefix}_mean": float(finite.mean()),
                       f"{prefix}_std": float(finite.std()),
                       f"{prefix}_min": float(finite.min()), f"{prefix}_max": float(finite.max())})
        result.update({f"{prefix}_p{p:02d}": float(np.percentile(finite, p)) for p in percentiles})
    return result


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def summarize(data, config, scope):
    cfg, iql = config["diagnostics"], config["iql"]
    count = len(data["action"])
    out = {"scope": scope, "samples": count, "q_scope": "expert_action",
           "n_step": iql["n_step"],
           "n_step_actual_mean": float(np.mean(data["n_step_actual"])) if "n_step_actual" in data else 1.0,
           "advantage_source": "min(target_q1,target_q2)-v",
           "weight_semantics": "loss_coefficient_not_measured_gradient_norm",
           "diagnostic_config": cfg}
    for key in ("q1", "q2", "target_q1", "target_q2", "v", "reward", "td_target",
                "damage_reward", "wrong_block_reward", "pressure_event",
                "pressure_spirit_loss", "pressure_reward", "far_distance_event",
                "horizontal_distance", "far_distance_penalty", "win_loss_reward",
                "n_step_reward", "n_step_discount", "n_step_actual"):
        if key in data:
            out.update(distribution(data[key], key))
    out.update(distribution(np.abs(data["q1"] - data["q2"]), "q_gap"))
    out.update(distribution(np.abs(data["target_q1"] - data["target_q2"]), "target_q_gap"))
    out.update(distribution(np.minimum(data["q1"], data["q2"]) - data["v"], "online_qmin_minus_v"))
    out.update(distribution(data["adv"], "adv", (1, 5, 10, 25, 50, 75, 90, 95, 99)))
    out.update(distribution(data["weight"], "weight", (50, 75, 90, 95, 99)))
    out.update(distribution(data["final_weight"], "final_weight", (50, 90, 95, 99)))
    out.update(distribution(data["effective_weight"], "effective_weight", (50, 90, 95, 99)))
    out.update(distribution(data["loss_coefficient"], "loss_coefficient", (50, 90, 95, 99)))
    log_weight = data["adv"].astype(np.float64) * iql["advantage_beta"]
    out.update(distribution(log_weight, "beta_adv"))
    clipped = log_weight >= math.log(iql["max_weight"])
    weighting = iql.get("actor_advantage_weighting", True)
    out.update(positive_adv_ratio=ratio((data["adv"] > 0).sum(), count),
               negative_adv_ratio=ratio((data["adv"] < 0).sum(), count),
               near_zero_adv_ratio=ratio((np.abs(data["adv"]) <= cfg["adv_near_zero_threshold"]).sum(), count),
               weight_clip_count=int(clipped.sum()) if weighting else 0,
               weight_clip_ratio=ratio(clipped.sum(), count) if weighting else 0.0,
               hypothetical_weight_clip_ratio=ratio(clipped.sum(), count),
               actor_advantage_weighting=weighting,
               reward_nonzero_ratio=ratio((data["reward"] != 0).sum(), count))
    if "pressure_event" in data:
        pressure = data["pressure_event"].astype(bool)
        out.update(pressure_event_count=int(pressure.sum()),
                   pressure_event_ratio=ratio(pressure.sum(), count),
                   pressure_spirit_loss_sum=float(data["pressure_spirit_loss"].sum()),
                   pressure_reward_sum=float(data["pressure_reward"].sum()))
    if "far_distance_event" in data:
        far = data["far_distance_event"].astype(bool)
        out.update(far_distance_event_count=int(far.sum()),
                   far_distance_event_ratio=ratio(far.sum(), count),
                   far_distance_penalty_sum=float(data["far_distance_penalty"].sum()))
    selected, changed = data["actor_selected"].astype(bool), data["changed"].astype(bool)
    # final_weight 是用户要求的乘积；effective_weight 还包含实际筛选和 Neutral 权重。
    final_sum, effective_sum = data["final_weight"].sum(), data["effective_weight"].sum()
    out.update(actor_selected_samples=int(selected.sum()),
               actor_updated_sample_ratio=ratio(data["actor_updated"].sum(), count),
               actor_loss_denominator_sum=float(data["denominator_weight"].sum()),
               final_weight_sum=float(final_sum), effective_weight_sum=float(effective_sum),
               applied_loss_coefficient_sum=float(data["applied_loss_coefficient"].sum()),
               keyframe_sample_ratio=ratio(changed.sum(), count),
               non_keyframe_sample_ratio=ratio((~changed).sum(), count),
               keyframe_final_weight_ratio=ratio(data["final_weight"][changed].sum(), final_sum),
               non_keyframe_final_weight_ratio=ratio(data["final_weight"][~changed].sum(), final_sum),
               keyframe_effective_weight_ratio=ratio(data["effective_weight"][changed].sum(), effective_sum))
    pair = data["switch_pair"].astype(bool)
    out.update(switch_pair_count=int(pair.sum()),
               expert_switch_rate=ratio(data["expert_switch"][pair].sum(), pair.sum()),
               predicted_switch_rate=ratio(data["predicted_switch"][pair].sum(), pair.sum()),
               expert_keyframe_rate=ratio(changed.sum(), data["history_eligible"].sum()))
    mapping = np.asarray(class_map(cfg))
    classes = mapping[data["action"]]
    predictions = data["prediction"]
    prediction_valid = predictions >= 0
    predicted_classes = mapping[np.maximum(predictions, 0)]
    out["classes"] = {}
    for name in sorted(set(mapping)):
        chosen = classes == name
        n = int(chosen.sum())
        final = data["final_weight"][chosen]
        effective = data["effective_weight"][chosen]
        stats = dict(sample_count=n, sample_fraction=ratio(n, count),
                     keyframe_count=int((chosen & changed).sum()),
                     keyframe_fraction=ratio((chosen & changed).sum(), n),
                     adv_mean=float(data["adv"][chosen].mean()) if n else None,
                     adv_median=float(np.median(data["adv"][chosen])) if n else None,
                     positive_adv_ratio=ratio((data["adv"][chosen] > 0).sum(), n),
                     iql_weight_mean=float(data["weight"][chosen].mean()) if n else None,
                     iql_weight_p95=float(np.percentile(data["weight"][chosen], 95)) if n else None,
                     keyframe_weight_mean=float(data["keyframe_weight"][chosen].mean()) if n else None,
                     final_weight_mean=float(final.mean()) if n else None,
                     final_weight_p95=float(np.percentile(final, 95)) if n else None,
                     final_weight_sum=float(final.sum()), final_weight_sum_fraction=ratio(final.sum(), final_sum),
                     effective_weight_sum=float(effective.sum()),
                     effective_weight_sum_fraction=ratio(effective.sum(), effective_sum),
                     loss_coefficient_sum=float(data["loss_coefficient"][chosen].sum()),
                     loss_coefficient_sum_fraction=ratio(data["loss_coefficient"][chosen].sum(), data["loss_coefficient"].sum()),
                     applied_loss_coefficient_sum=float(data["applied_loss_coefficient"][chosen].sum()),
                     actor_selected_count=int((chosen & selected).sum()),
                     mean_immediate_reward=float(data["reward"][chosen].mean()) if n else None,
                     mean_td_target=float(data["td_target"][chosen].mean()) if n else None,
                     mean_q=float(np.minimum(data["q1"], data["q2"])[chosen].mean()) if n else None)
        out["classes"][name] = stats
        out[f"expert_{name.lower()}_ratio"] = ratio(n, count)
        out[f"predicted_{name.lower()}_ratio"] = ratio(((predicted_classes == name) & prediction_valid).sum(), prediction_valid.sum())
    for unknown in ("flight", "defense", "skill", "spell", "walk"):
        out.setdefault(f"expert_{unknown}_ratio", None)
        out.setdefault(f"predicted_{unknown}_ratio", None)
    # 攻击按 A/B/C 按键定义，包含 D+A 等组合，与可配置类别名称解耦。
    attack = (data["action"] % 16 & 13) != 0
    predicted_attack = (np.maximum(predictions, 0) % 16 & 13) != 0
    out.update(expert_attack_ratio=ratio(attack.sum(), count),
               predicted_attack_ratio=ratio((predicted_attack & prediction_valid).sum(), prediction_valid.sum()),
               attack_final_weight_ratio=ratio(data["final_weight"][attack].sum(), final_sum),
               attack_effective_weight_ratio=ratio(data["effective_weight"][attack].sum(), effective_sum),
               attack_adv_mean=float(data["adv"][attack].mean()) if attack.any() else None,
               nonattack_adv_mean=float(data["adv"][~attack].mean()) if (~attack).any() else None)
    out["warnings"] = warnings(out, cfg["warnings"])
    out["assessment"] = assessment(out, cfg["warnings"])
    return clean(out)


def assessment(row, limits):
    """逐项给出有数据支持的观测；相关性不作为 Critic 偏差的因果证明。"""
    attack, other = row["attack_adv_mean"], row["nonattack_adv_mean"]
    gap = attack - other if attack is not None and other is not None else None
    return {
        "1_attack_advantage": {"attack_mean": attack, "nonattack_mean": other, "gap": gap},
        "2_exponential_amplification": {"beta_adv_std": row["beta_adv_std"],
                                       "weight_p50": row["weight_p50"], "weight_p99": row["weight_p99"]},
        "3_attack_final_share": {"sample": row["expert_attack_ratio"], "keyframe_x_iql": row["attack_final_weight_ratio"],
                                 "after_filter_and_category": row["attack_effective_weight_ratio"]},
        "4_clipping": {"ratio": row["weight_clip_ratio"], "threshold": limits["weight_clip_ratio"]},
        "5_critic_agreement": {"online_gap_mean": row["q_gap_mean"], "target_gap_mean": row["target_q_gap_mean"],
                               "note": "双Q一致不等于价值正确；单次报告不能证明长期稳定"},
        "6_collapse": {"adv_std": row["adv_std"], "near_zero": row["near_zero_adv_ratio"]},
        "7_policy_drift": {"expert_attack": row["expert_attack_ratio"], "predicted_attack": row["predicted_attack_ratio"],
                           "expert_switch": row["expert_switch_rate"], "predicted_switch": row["predicted_switch_rate"],
                           "note": "专家历史条件下的离线预测；持续漂移需同一probe_id多步比较"},
        "8_cause": "结合类别Q/Advantage差异与样本→乘积→有效系数占比判断；本报告不能单独证明Critic value bias的因果关系",
    }


def warnings(row, limits):
    found = []
    def above(key, limit, message):
        if row.get(key) is not None and row[key] > limits[limit]:
            found.append(f"{message}：{key}={row[key]:.5g}，阈值={limits[limit]:g}")
    above("weight_clip_ratio", "weight_clip_ratio", "指数权重大量截顶")
    for prefix in ("", "target_"):
        above(prefix + "q_gap_mean", "q_gap_mean", "双 Q 均值分歧较大")
        above(prefix + "q_gap_p95", "q_gap_p95", "双 Q 尾部分歧较大")
    for key in ("q1_std", "q2_std"):
        above(key, "q_std", "Q 波动较大")
    if row.get("adv_std") is not None and row["adv_std"] < limits["adv_std_min"]:
        found.append("Advantage std 较小：检查优势塌缩或价值网络尚未学成")
    above("near_zero_adv_ratio", "near_zero_ratio", "Advantage 接近零比例较高")
    above("beta_adv_std", "beta_adv_std", "beta×Advantage 离散程度较大，可能放大权重差")
    for name, stats in row["classes"].items():
        share = stats["effective_weight_sum_fraction"]
        if (stats["sample_count"] >= limits["class_min_samples"] and share is not None
                and share > limits["class_weight_share"]
                and share > stats["sample_fraction"] * limits["class_share_amplification"]):
            found.append(f"{name} 有效系数占比异常：样本 {stats['sample_fraction']:.1%} → 权重 {share:.1%}")
    for prefix, threshold in (("attack", "attack_ratio_excess"), ("switch", "switch_ratio_excess")):
        suffix = "ratio" if prefix == "attack" else "rate"
        predicted, expert = row.get(f"predicted_{prefix}_{suffix}"), row.get(f"expert_{prefix}_{suffix}")
        if predicted is not None and expert is not None and predicted - expert > limits[threshold]:
            found.append(f"预测 {prefix} 高于专家：{predicted:.1%} vs {expert:.1%}")
    if row.get("attack_adv_mean") is not None and row.get("nonattack_adv_mean") is not None:
        if row["attack_adv_mean"] - row["nonattack_adv_mean"] > limits["attack_adv_gap"]:
            found.append("攻击按键组 Advantage 系统性高于非攻击组；需结合固定状态轨迹判断")
    if any(value for key, value in row.items() if key.endswith("nonfinite_count")):
        found.append("诊断张量存在 NaN/Inf；分位数只统计有限值，结论需谨慎")
    return found


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def append(path, row):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(clean(row), ensure_ascii=False, allow_nan=False) + "\n")


def report(row):
    def fmt(key):
        value = row.get(key)
        return "N/A" if value is None else f"{value:.5g}"
    lines = ["================ IQL Health Report ================", f"step={row['step']} scope={row['scope']} samples={row['samples']}"]
    for title, keys in (
        ("Q1 mean/std", ("q1_mean", "q1_std")), ("Q2 mean/std", ("q2_mean", "q2_std")),
        ("Q gap mean/p95/max", ("q_gap_mean", "q_gap_p95", "q_gap_max")),
        ("V mean/std", ("v_mean", "v_std")),
        ("Advantage mean/std/positive/near-zero", ("adv_mean", "adv_std", "positive_adv_ratio", "near_zero_adv_ratio")),
        ("IQL weight mean/p95/max/clip", ("weight_mean", "weight_p95", "weight_max", "weight_clip_ratio")),
        ("Keyframe sample/final/effective", ("keyframe_sample_ratio", "keyframe_final_weight_ratio", "keyframe_effective_weight_ratio")),
        ("Attack expert/predicted/effective", ("expert_attack_ratio", "predicted_attack_ratio", "attack_effective_weight_ratio")),
        ("Switch expert/predicted", ("expert_switch_rate", "predicted_switch_rate")),
    ):
        lines.append(title + ": " + " / ".join(fmt(key) for key in keys))
    lines.append("Actor 类别系数（不是直接测量的梯度范数）：")
    for name, stats in row["classes"].items():
        lines.append(f"  {name}: samples={stats['sample_fraction']} final={stats['final_weight_sum_fraction']} effective={stats['effective_weight_sum_fraction']} adv={stats['adv_mean']} Q={stats['mean_q']} reward={stats['mean_immediate_reward']} TD={stats['mean_td_target']}")
        lines.append(f"    expert={row.get('expert_' + name.lower() + '_ratio')} predicted={row.get('predicted_' + name.lower() + '_ratio')}")
    lines.extend(["Warnings:", *["  " + item for item in row["warnings"]], "==================================================="])
    lines.insert(-1, "诊断观测：" + json.dumps(row["assessment"], ensure_ascii=False))
    return "\n".join(lines)
