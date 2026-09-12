import pytest
import torch

from soku_iql.actor_sampling import build_actor_mask
from soku_iql.bc_core.action_space import NEUTRAL_ACTION_ID


def test_cap_uses_selected_denominator_and_preserves_inputs():
    actions = torch.tensor([[64]*9 + [65]*3])
    valid = torch.ones_like(actions, dtype=torch.bool)
    original = valid.clone()
    mask, stats = build_actor_mask(actions, valid, True, .25)
    assert mask.shape == actions.shape
    assert int(mask.sum()) == 4
    assert int((mask & (actions == NEUTRAL_ACTION_ID)).sum()) == 1
    assert mask[actions != 64].all()
    assert torch.equal(valid, original)
    assert stats['actor_neutral_fraction_after'] == .25


@pytest.mark.parametrize('fraction', [0, .25, .5, .9])
def test_all_neutral_is_empty_below_one(fraction):
    actions = torch.full((2, 4), 64)
    mask, stats = build_actor_mask(actions, torch.ones_like(actions).bool(), True, fraction)
    assert not mask.any()
    assert stats['actor_neutral_fraction_after'] is None


def test_disabled_and_one_keep_original_mask():
    actions = torch.tensor([[64, 64, 65]])
    valid = torch.tensor([[True, False, True]])
    for enabled, fraction in [(False, .25), (True, 1)]:
        mask, _ = build_actor_mask(actions, valid, enabled, fraction)
        assert torch.equal(mask, valid)


def test_padding_excluded_and_crouch_not_neutral():
    actions = torch.tensor([[64, 64, 16, 65, 66]])
    valid = torch.tensor([[True, False, True, True, True]])
    mask, stats = build_actor_mask(actions, valid, True, .25)
    assert torch.equal(mask, valid)
    assert stats['actor_candidate_samples'] == 4


def test_decimal_cap_and_rng_repeatability():
    actions = torch.tensor([[64]*8 + [65]*7])
    valid = torch.ones_like(actions).bool()
    torch.manual_seed(7)
    first, stats = build_actor_mask(actions, valid, True, .3)
    torch.manual_seed(7)
    second, _ = build_actor_mask(actions, valid, True, .3)
    assert torch.equal(first, second)
    assert stats['actor_neutral_after'] == 3
    assert stats['actor_neutral_fraction_after'] == .3
