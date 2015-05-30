import unittest
import game2p3s as g
import numpy as np

LAZY_STRATEGY = range(1, 10)

LAZY_STRATEGY_HALF_GAME = g.HalfGameArray(LAZY_STRATEGY)
LAZY_STRATEGY_GAME = g.GameArray(LAZY_STRATEGY * 2)


def random_mirror_game():
    half_game = g.random_half_game()
    return g.GameArray([half_game] * 2)


class TestHalfGame(unittest.TestCase):
    def setUp(self):
        self.lazy_half = LAZY_STRATEGY_HALF_GAME.copy()

    def test_lazy_strategy_standard(self):
        standard_is = self.lazy_half.standard.view(np.ndarray)
        should_be = np.array([[9, 7, 8], [3, 5, 4], [6, 2, 1]])
        self.assertTrue((standard_is == should_be).all())

    def test_multiple_random_standardizations(self, how_many=100):
        for i in range(how_many):
            half = g.random_half_game()
            hs = half.standard
            self.assertEqual(hs[0, 0], 9)
            self.assertEqual(hs[1, 1], np.amax(hs[1:, 1:]))


class TestGame(unittest.TestCase):
    def setUp(self):
        self.lazy = LAZY_STRATEGY_GAME.copy()

    def test_lazy_strategy_standard(self):
        standard_is = self.lazy.standard.view(np.ndarray)
        should_be = np.array([[9, 7, 8], [3, 5, 4], [6, 2, 1]] * 2).reshape((2,3,3))
        self.assertTrue((standard_is == should_be).all())

    def test_random_mirror_game_standard(self):
        rmg = random_mirror_game().standard

        for player in range(2):
            self.assertEqual(rmg[player, 0, 0], 9)
            self.assertEqual(rmg[player, 1, 1], np.amax(rmg[0, 1:, 1:]))

        player_0_set = set(rmg.player(0).flatten().tolist())
        player_1_set = set(rmg.player(1).flatten().tolist())

        self.assertEqual(player_0_set, set(range(1, 10)))
        self.assertEqual(player_0_set, player_1_set)

    def test_multiple_random_standardizations(self, how_many=100):
        for i in range(how_many):
            game = g.random_game().standard
            self.assertEqual(game[0, 0, 0], 9)
            self.assertEqual(game[0, 1, 1], np.amax(game[0, 1:, 1:]))


if __name__ == '__main__':
    unittest.main()
