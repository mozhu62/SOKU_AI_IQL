import torch

from soku_iql.actor_weighting import weighted_actor_loss


def test_neutral_weight_and_detached_advantage():
    ce = torch.tensor([2., 1.], requires_grad=True)
    advantage = torch.tensor([1., 1.], requires_grad=True)
    loss = weighted_actor_loss(ce, advantage, torch.tensor([64, 65]), .25)
    torch.testing.assert_close(loss, torch.tensor(1.2))
    loss.backward()
    torch.testing.assert_close(ce.grad, torch.tensor([.2, .8]))
    assert advantage.grad is None


def test_one_matches_original_iql_mean():
    ce, advantage = torch.tensor([2., 3.]), torch.tensor([4., 5.])
    result = weighted_actor_loss(ce, advantage, torch.tensor([64, 65]), 1.)
    torch.testing.assert_close(result, (ce * advantage).mean())


def test_all_neutral_still_has_gradient():
    ce = torch.tensor([2., 4.], requires_grad=True)
    result = weighted_actor_loss(ce, torch.ones(2), torch.tensor([64, 64]), .25)
    torch.testing.assert_close(result, ce.mean())
    result.backward()
    torch.testing.assert_close(ce.grad, torch.tensor([.5, .5]))
