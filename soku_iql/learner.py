from __future__ import annotations

import math
import time

import torch
from torch.nn import functional as F

from .models import IQLNetworks
from .actor_sampling import build_actor_mask
from .actor_weighting import weighted_actor_loss


def tensor_batch(value, device):
    if isinstance(value, dict):
        return {key: tensor_batch(item, device) for key, item in value.items()}
    return torch.as_tensor(value).to(device, non_blocking=True)


def expectile_loss(diff, expectile):
    return torch.where(diff > 0, expectile, 1 - expectile) * diff.square()


def advantage_weights(advantage, beta, maximum):
    # 先截 log 权重，避免 exp 溢出；不除以权重之和，保持 IQL 的加权均值定义。
    return (advantage.detach() * beta).clamp(max=math.log(maximum)).exp()


def td_target(rewards, terminal, next_value, gamma):
    return rewards + gamma * torch.where(terminal, 0, next_value)


def batch_td_target(batch, values, gamma):
    if "bootstrap_index" in batch:
        following = values.gather(1, batch["bootstrap_index"].long())
        return td_target(batch["n_step_reward"], batch["n_step_terminal"], following, batch["n_step_discount"])
    # 保留旧单步测试/调用者兼容；正式 Dataset 始终提供完整 N-step 字段。
    return td_target(batch["reward"], batch["terminal"], values[:, 1:], gamma)


class Learner:
    def __init__(self, config, actor_state=None):
        self.config = config
        request = config["training"]["device"]
        self.device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if request == "auto" else request)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("指定 CUDA 但当前不可用")
        torch.set_num_threads(config["training"]["cpu_threads"])
        self.networks = IQLNetworks(config["model"], actor_state).to(self.device)
        self.amp = config["training"]["amp"] and self.device.type == "cuda"
        q = config["iql"]
        params = dict(actor=list(self.networks.actor.parameters()),
                      critic=list(self.networks.q1.parameters()) + list(self.networks.q2.parameters()),
                      value=list(self.networks.value.parameters()))
        self.params = params
        self.optimizers = {key: torch.optim.AdamW(values, lr=q[f"{key}_lr"], weight_decay=q["weight_decay"])
                           for key, values in params.items()}
        self.scalers = {key: torch.amp.GradScaler("cuda", enabled=self.amp) for key in params}

    def forward(self, model, batch):
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.amp):
            return model(batch["observation"], self.config["training"]["burn_in"], batch["burn_lengths"]).float()

    def optimize(self, key, loss):
        if not torch.isfinite(loss):
            raise FloatingPointError(f"{key} loss 非有限，停止训练且不覆盖 checkpoint")
        optimizer, scaler = self.optimizers[key], self.scalers[key]
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(self.params[key], self.config["training"]["max_grad_norm"])
        if not self.amp and not torch.isfinite(norm):
            raise FloatingPointError(f"{key} 梯度非有限，未执行 optimizer.step")
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        return scaler.get_scale() >= old_scale, float(norm)

    @torch.no_grad()
    def q_reference(self, batch):
        n = self.networks
        action = batch["action"][..., None]
        first = self.forward(n.target_q1, batch)[:, :action.shape[1]].gather(-1, action).squeeze(-1)
        second = self.forward(n.target_q2, batch)[:, :action.shape[1]].gather(-1, action).squeeze(-1)
        return torch.minimum(first, second)

    def train_batch(self, raw, step, diagnostics=False):
        started = time.perf_counter()
        batch = tensor_batch(raw, self.device)
        mask, action = batch["mask"].bool(), batch["action"].long()
        if not mask.any():
            raise ValueError("批次没有有效 transition")
        n, cfg = self.networks, self.config["iql"]
        for model in (n.actor, n.q1, n.q2, n.value):
            model.train()
        q_ref = self.q_reference(batch)
        value = self.forward(n.value, batch)[:, :action.shape[1], 0]
        v_loss = expectile_loss(q_ref - value, cfg["expectile"])[mask].mean()
        v_updated, v_grad = self.optimize("value", v_loss)
        with torch.no_grad():
            new_value = self.forward(n.value, batch)[..., 0]
            advantage = q_ref - new_value[:, :action.shape[1]]
            weights = advantage_weights(advantage, cfg["advantage_beta"], cfg["max_weight"])
            # Q 回归真实轨迹 N 步回报及后继 V，不做 argmax 或跨终局 bootstrap。
            target = batch_td_target(batch, new_value, cfg["gamma"])
        actor_updated, actor_grad, actor_loss = False, None, None
        sampling = self.config.get("actor_sampling", {})
        weighting = self.config.get("actor_weighting", {})
        weighting_enabled = weighting.get("enabled", False)
        if weighting_enabled and sampling.get("enabled", False):
            raise ValueError("Actor 筛选与类别加权不能同时启用")
        neutral_weight = weighting.get("neutral_weight", .25) if weighting_enabled else 1.0
        actor_mask, sampling_stats = build_actor_mask(action, mask, **sampling)
        actor_skip_reason = "warmup" if step < cfg["actor_warmup_steps"] else "no_samples" if not sampling_stats["actor_samples"] else None
        if actor_skip_reason is None:
            logits = self.forward(n.actor, batch)[:, :action.shape[1]]
            ce = F.cross_entropy(logits[actor_mask], action[actor_mask], reduction="none")
            actor_loss = weighted_actor_loss(ce, weights[actor_mask], action[actor_mask], neutral_weight)
            actor_updated, actor_grad = self.optimize("actor", actor_loss)
            if not actor_updated:
                actor_skip_reason = "amp"
        else:
            self.optimizers["actor"].zero_grad(set_to_none=True)
        q1 = self.forward(n.q1, batch)[:, :action.shape[1]].gather(-1, action[..., None]).squeeze(-1)
        q2 = self.forward(n.q2, batch)[:, :action.shape[1]].gather(-1, action[..., None]).squeeze(-1)
        q_loss = ((q1 - target).square() + (q2 - target).square())[mask].mean()
        q_updated, q_grad = self.optimize("critic", q_loss)
        if q_updated:
            n.update_targets(cfg["target_tau"])
        result = dict(samples=int(mask.sum()), value_loss=float(v_loss.detach()), q_loss=float(q_loss.detach()),
                      actor_loss=float(actor_loss.detach()) if actor_loss is not None else None,
                      actor_updated=actor_updated, critic_updated=q_updated, value_updated=v_updated,
                      actor_gradient_norm=actor_grad, critic_gradient_norm=q_grad, value_gradient_norm=v_grad,
                      advantage_mean=float(advantage[mask].mean()), weight_mean=float(weights[mask].mean()),
                      weight_max=float(weights[mask].max()), warmup=step < cfg["actor_warmup_steps"])
        if diagnostics:
            result.update(self.validate_batch(raw))
        result.update(sampling_stats, actor_skip_reason=actor_skip_reason,
                      n_step=cfg.get("n_step", 1),
                      n_step_actual_mean=float(batch["n_step_steps"][mask].float().mean()) if "n_step_steps" in batch else 1.0,
                      actor_weighting_enabled=weighting_enabled, actor_neutral_weight=neutral_weight,
                      actor_optimized_samples=sampling_stats["actor_samples"] if actor_updated else 0)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        result["optimization_seconds"] = time.perf_counter() - started
        return result

    @torch.no_grad()
    def validate_batch(self, raw):
        b = tensor_batch(raw, self.device)
        for model in (self.networks.actor, self.networks.q1, self.networks.q2, self.networks.value):
            model.eval()
        mask, action = b["mask"].bool(), b["action"].long()
        logits = self.forward(self.networks.actor, b)[:, :action.shape[1]][mask]
        labels = action[mask]
        q1 = self.forward(self.networks.q1, b)[:, :action.shape[1]].gather(-1, action[..., None]).squeeze(-1)[mask]
        q2 = self.forward(self.networks.q2, b)[:, :action.shape[1]].gather(-1, action[..., None]).squeeze(-1)[mask]
        v = self.forward(self.networks.value, b)[..., 0]
        target = batch_td_target(b, v, self.config["iql"]["gamma"])[mask]
        if not all(torch.isfinite(x).all() for x in (logits, q1, q2, v, target)):
            raise FloatingPointError("验证输出含 NaN/Inf")
        prediction = logits.argmax(-1)
        previous = b["previous_action"][mask]
        eligible = (previous >= 0) & (previous < logits.shape[-1])
        change = eligible & (previous != labels)
        error = torch.minimum(q1, q2) - target
        return dict(samples=int(mask.sum()), nll=float(F.cross_entropy(logits, labels)),
                    top1=float((prediction == labels).float().mean()),
                    top5=float((logits.topk(5, -1).indices == labels[:, None]).any(-1).float().mean()),
                    change_count=int(change.sum()), change_correct=int(((prediction == labels) & change).sum()),
                    previous_eligible=int(eligible.sum()), previous_correct=int(((previous == labels) & eligible).sum()),
                    td_mse=float(error.square().mean()), td_mae=float(error.abs().mean()),
                    q_mean=float(torch.minimum(q1, q2).mean()), v_mean=float(v[:, :action.shape[1]][mask].mean()),
                    target_sum=float(target.sum()), target_square_sum=float(target.double().square().sum()),
                    residual_sum=float(error.sum()), residual_square_sum=float(error.double().square().sum()))
