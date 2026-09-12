import copy

import numpy as np
import pytest
import torch

from soku_iql import checkpoint
from soku_iql.bc_core import checkpoint as bc_checkpoint
from soku_iql.bc_core.config import DEFAULTS
from soku_iql.config import TRAINING, IQL, REWARD, validate_resume
from soku_iql.dataset import IQLReplayStore
from soku_iql.learner import Learner, advantage_weights, expectile_loss, td_target
from soku_iql.runtime import aggregate


def config():
    bc = copy.deepcopy(DEFAULTS)
    cfg = dict(seed=42, model=bc["model"], data=bc["data"], training=copy.deepcopy(TRAINING),
               iql=copy.deepcopy(IQL), reward=copy.deepcopy(REWARD))
    cfg["training"].update(device="cpu", amp=False, cpu_threads=1, batch_size=1, sequence_length=2)
    cfg["iql"]["actor_warmup_steps"] = 1
    return bc, cfg


def batch():
    shape = (1, 34)
    obs = dict(state_continuous=torch.zeros(*shape, 18), tactical_state=torch.zeros(*shape, 9),
               state_categorical=torch.zeros(*shape, 5, dtype=torch.long),
               previous_joint_action_id=torch.full(shape, 64, dtype=torch.long),
               previous_action_duration=torch.zeros(*shape, 1), state_optional_continuous=torch.zeros(*shape, 4),
               state_optional_mask=torch.zeros(*shape, 4, dtype=torch.bool))
    for side in ("self", "opponent"):
        obs.update({f"{side}_object_numerical": torch.zeros(*shape, 3, 8),
                    f"{side}_object_categorical": torch.zeros(*shape, 3, 2, dtype=torch.long),
                    f"{side}_object_mask": torch.zeros(*shape, 3, dtype=torch.bool),
                    f"{side}_skill_categorical": torch.zeros(*shape, 4, dtype=torch.long),
                    f"{side}_skill_mask": torch.zeros(*shape, 4, dtype=torch.bool)})
    return dict(observation=obs, burn_lengths=torch.tensor([31]), action=torch.tensor([[64, 65]]),
                previous_action=torch.tensor([[64, 64]]), mask=torch.tensor([[True, True]]),
                terminal=torch.tensor([[False, True]]), reward=torch.tensor([[1., -1.]]))


def test_all_neutral_skips_actor_but_updates_qv():
    _, cfg = config()
    cfg['actor_sampling'] = dict(enabled=True, neutral_max_fraction=.25)
    learner = Learner(cfg)
    raw = batch()
    raw['action'].fill_(64)
    before = {key: value.clone() for key, value in learner.networks.actor.state_dict().items()}
    result = learner.train_batch(raw, step=1)
    assert result['actor_skip_reason'] == 'no_samples'
    assert result['actor_loss'] is None and not result['actor_updated']
    assert result['actor_optimized_samples'] == 0
    assert result['critic_updated'] and result['value_updated']
    assert result['samples'] == 2
    for key, value in learner.networks.actor.state_dict().items():
        assert torch.equal(before[key], value)


def test_loss_definitions():
    diff = torch.tensor([-2., 2.])
    torch.testing.assert_close(expectile_loss(diff, .7), torch.tensor([1.2, 2.8]))
    advantage = torch.tensor([0., 1000.], requires_grad=True)
    weights = advantage_weights(advantage, 3., 100.)
    torch.testing.assert_close(weights, torch.tensor([1., 100.]))
    assert not weights.requires_grad
    target = td_target(torch.tensor([1., 1.]), torch.tensor([False, True]), torch.tensor([10., 1000.]), .9)
    torch.testing.assert_close(target, torch.tensor([10., 1.]))


def test_real_next_state_remains_in_same_sequence():
    store = IQLReplayStore.__new__(IQLReplayStore)
    store.split = {"train": ["one"]}
    store.info = {"one": {"transitions": 2}}
    shard = dict(segments=np.array([[2, 4]]), segment_cumulative=np.array([2]),
                 joint_action_id=np.arange(6), previous_expert_action_id=np.arange(6),
                 iql_reward=np.arange(6, dtype=np.float32), iql_terminal=np.array([0, 0, 0, 1, 0, 0], bool))
    store.get = lambda _: shard
    store.observation = lambda _, indices: {"frame": indices}
    cfg = dict(batch_size=16, sequence_length=3, burn_in=31, replays_per_batch=1)
    result = store.sample(np.random.default_rng(5), cfg)
    for row in range(16):
        frames = result["observation"]["frame"][row, 31:]
        for column in np.flatnonzero(result["mask"][row]):
            assert frames[column + 1] == frames[column] + 1
            assert result["action"][row, column] == frames[column]
        assert frames.max() <= 4


def test_warmup_isolation_resume_and_bc_export(tmp_path):
    bc, cfg = config()
    learner = Learner(cfg)
    initial = {k: v.clone() for k, v in learner.networks.actor.state_dict().items()}
    # BC 参数严格迁移，评分网络与 Actor 不共享 Parameter。
    transferred = Learner(cfg, initial)
    assert all(torch.equal(v, initial[k]) for k, v in transferred.networks.actor.state_dict().items())
    actor_ids = {id(p) for p in learner.networks.actor.parameters()}
    assert not actor_ids & {id(p) for p in learner.networks.q1.parameters()}
    assert all(not p.requires_grad for p in learner.networks.target_q1.parameters())
    result = learner.train_batch(batch(), 0)
    assert not result["actor_updated"]
    assert all(torch.equal(v, initial[k]) for k, v in learner.networks.actor.state_dict().items())
    result = learner.train_batch(batch(), 1)
    assert result["actor_updated"]
    assert any(not torch.equal(v, initial[k]) for k, v in learner.networks.actor.state_dict().items())
    path = tmp_path / "last.pt"
    checkpoint.save(path, learner, bc, {}, "fixed", 2, 4, 1, 1.0, {})
    package = checkpoint.load(path)
    assert package["step"] == 2
    restored = Learner(cfg)
    restored.networks.load_state_dict(package["networks"], strict=True)
    for key in restored.optimizers:
        restored.optimizers[key].load_state_dict(package["optimizers"][key])
    exported = tmp_path / "actor_bc.pt"
    checkpoint.export_actor(exported, learner, bc, {}, "fixed", 2, 4, 1, {})
    loaded = bc_checkpoint.load(exported)
    assert loaded["provenance"]["trained_algorithm"] == "iql"
    # 原 BC 的 best 是 None 或指标字典，不能写 float("inf") 导致后续验证索引失败。
    assert loaded["best"] is None
    transferred.networks.actor.load_state_dict(loaded["model"], strict=True)
    learner.networks.actor.eval()
    transferred.networks.actor.eval()
    b = batch()
    with torch.no_grad():
        torch.testing.assert_close(learner.forward(learner.networks.actor, b),
                                   transferred.forward(transferred.networks.actor, b), rtol=0, atol=0)


def test_constant_targets_do_not_report_fake_ev():
    row = dict(samples=2, nll=1., top1=.5, top5=1., td_mse=0., td_mae=0., q_mean=1., v_mean=1.,
               target_sum=2., target_square_sum=2., residual_sum=0., residual_square_sum=0.,
               change_count=0, change_correct=0, previous_eligible=2, previous_correct=2)
    result = aggregate([row])
    assert result["ev"] is None and result["change_top1"] is None


def test_resume_preserves_sampling_and_validation_baseline():
    _, cfg = config()
    original = copy.deepcopy(cfg)
    validate_resume(cfg, original)
    for key in ("batch_size", "replays_per_batch", "validation_batches", "burn_in", "sequence_length"):
        changed = copy.deepcopy(original)
        changed["training"][key] += 1
        with pytest.raises(ValueError, match=key):
            validate_resume(changed, original)


def test_live_adapter_reads_actor_from_full_iql(tmp_path):
    from soku_iql.live.policy_checkpoint import load as live_load
    bc, cfg = config()
    learner = Learner(cfg)
    path = tmp_path / "iql.pt"
    checkpoint.save(path, learner, bc, {}, "fixed", 0, 0, 0, 1.0, {})
    package = live_load(path)
    assert package["provenance"]["trained_algorithm"] == "iql"
    original = learner.networks.actor.state_dict()
    assert package["model"].keys() == original.keys()
    assert all(torch.equal(v, package["model"][k]) for k, v in original.items())
    assert "networks" not in package and "optimizers" not in package
