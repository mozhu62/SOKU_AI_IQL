import unittest
import numpy as np
from soku_iql.dataset import ko_outcomes


class KORewardTests(unittest.TestCase):
    def test_first_ko_only(self):
        result = ko_outcomes(np.array([50, 50, 50]), np.array([10, 0, 0]),
                             np.array([True, True, False]), np.array([True, True, False]))
        np.testing.assert_array_equal(result, [1, 0, 0])

    def test_loss(self):
        result = ko_outcomes(np.array([10, 0]), np.array([50, 50]),
                             np.array([True, False]), np.array([True, False]))
        np.testing.assert_array_equal(result, [-1, 0])

    def test_unknown_or_invalid(self):
        for own, enemy, valid, terminal in [([10, 0], [10, 0], True, True),
                                            ([10, 10], [10, 0], False, True),
                                            ([10, 10], [10, 0], True, False)]:
            result = ko_outcomes(np.array(own), np.array(enemy), np.array([valid, False]),
                                 np.array([terminal, False]))
            np.testing.assert_array_equal(result, [0, 0])
