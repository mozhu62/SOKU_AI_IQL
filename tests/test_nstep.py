import numpy as np
import torch

from soku_iql.dataset import n_step_returns
from soku_iql.learner import batch_td_target


def test_discounted_rewards_and_terminal_stop():
    rewards = np.array([1., 2., 3., 100.], np.float32)
    returns, discounts, done, steps = n_step_returns(
        rewards, np.array([False, False, True, False]), np.array([[0, 1]]), np.array([[3]]), 3, .9)
    np.testing.assert_allclose(returns, [[5.23, 4.7]], rtol=1e-6)
    np.testing.assert_allclose(discounts, [[.9**3, .9**2]], rtol=1e-6)
    assert done.all()
    np.testing.assert_array_equal(steps, [[3, 2]])


def test_fragment_end_does_not_consume_next_fragment():
    returns, discounts, done, steps = n_step_returns(
        np.array([1., 2., 999.]), np.zeros(3, bool), np.array([[1, 2]]), np.array([[2]]), 3, .9)
    np.testing.assert_allclose(returns, [[2., 0.]])
    np.testing.assert_array_equal(steps, [[1, 0]])
    assert not done.any()
    np.testing.assert_allclose(discounts, [[.9, 1.]])


def test_bootstrap_gathers_actual_successor_and_masks_terminal():
    batch = dict(bootstrap_index=torch.tensor([[3, 2]]), n_step_reward=torch.tensor([[1., 2.]]),
                 n_step_terminal=torch.tensor([[False, True]]), n_step_discount=torch.tensor([[.729, .9]]))
    target = batch_td_target(batch, torch.tensor([[0., 100., 1000., 10.]]), .99)
    torch.testing.assert_close(target, torch.tensor([[8.29, 2.]]))


def test_one_step_equivalence():
    returns, discounts, done, steps = n_step_returns(np.array([2., 3., 0.]),
        np.array([False, True, False]), np.array([[0, 1]]), np.array([[2]]), 1, .99)
    np.testing.assert_allclose(returns, [[2., 3.]])
    np.testing.assert_allclose(discounts, [[.99, .99]])
    np.testing.assert_array_equal(done, [[False, True]])
    np.testing.assert_array_equal(steps, [[1, 1]])
