import unittest
import torch
from soku_iql.keyframes import build_changepoint_mask
from soku_iql.actor_weighting import weighted_actor_loss


class KeyframeActorTests(unittest.TestCase):
    def test_boundaries(self):
        action = torch.tensor([[10, 10, 20, 20, 5]])
        previous = torch.tensor([[144, 10, 10, 144, 20]])
        changed, eligible = build_changepoint_mask(action, torch.ones_like(action, dtype=torch.bool), previous)
        self.assertEqual(changed.tolist(), [[False, False, True, False, True]])
        self.assertEqual(eligible.tolist(), [[False, True, True, False, True]])

    def test_weighted_reduction(self):
        loss = weighted_actor_loss(torch.tensor([1., 2.]), torch.ones(2), torch.tensor([10, 20]),
                                   keyframe_weights=torch.tensor([1., 4.]))
        self.assertAlmostEqual(loss.item(), 1.8, places=6)

    def test_unit_weights(self):
        ce = torch.tensor([1., 2.])
        loss = weighted_actor_loss(ce, torch.ones(2), torch.tensor([10, 20]), keyframe_weights=torch.ones(2))
        torch.testing.assert_close(loss, ce.mean())
