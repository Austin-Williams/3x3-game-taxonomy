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
            half = g.random_half_game().standard
            self.assertEqual(half[0, 0], np.amax(half))
            self.assertEqual(half[1, 1], np.amax(half[1:, 1:]))

    def test_multiple_random_standardizations_for_repeats(self, how_many=100):
        for i in range(how_many):
            half = g.random_half_game().standard
            self.assertEqual(set(half.flatten().tolist()), set(range(1,10)))

    def test_row_and_column_sets_do_not_change(self, how_many=100):
        def get_row_column_sets(half_game):
            rows = sorted([sorted(list(half[x])) for x in range(3)])
            columns = sorted([sorted(list(half[:, x]))
                           for x in range(3)])

            return tuple(rows), tuple(columns)

        for i in range(how_many):
            half = g.random_half_game()

            rc_before = get_row_column_sets(half)
            rc_after = get_row_column_sets(half.standard)

            self.assertEqual(rc_before, rc_after)


class TestGame(unittest.TestCase):
    def setUp(self):
        self.lazy = LAZY_STRATEGY_GAME.copy()

    def test_lazy_strategy_standard(self):
        standard_is = self.lazy.standard.view(np.ndarray)
        should_be = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]] * 2).reshape((2,3,3))
        self.assertTrue((standard_is == should_be).all())

    def test_random_mirror_game_standard(self):
        rmg = random_mirror_game().standard

        for player in range(2):
            self.assertEqual(rmg[player, 0, 0], np.amax(rmg[player]))
            self.assertEqual(rmg[player, 1, 1], np.amax(rmg[player, 1:, 1:]))

        player_0_strategy = rmg.player(0).flatten().tolist()
        player_1_strategy = rmg.player(1).flatten().tolist()
        player_0_set = set(player_0_strategy)

        self.assertEqual(player_0_set, set(range(1, 10)))
        self.assertEqual(player_0_strategy, player_1_strategy)

    def test_multiple_random_standardizations(self, how_many=100):
        for i in range(how_many):
            game = g.random_game().standard
            self.assertEqual(game[0, 0, 0], 9)
            self.assertEqual(game[0, 1, 1], np.amax(game[0, 1:, 1:]))

    def test_player_lockstep_during_standardizations(self, how_many=100):
        for i in range(how_many):
            game = g.random_game()
            std = game.standard

            p0 = game.player(0)
            p1 = game.player(1)
            sp0 = std.player(0)
            sp1 = std.player(1)

            for i in range(1,10):
                indices = np.where(p0.view(np.ndarray)==i)
                sindices = np.where(sp0.view(np.ndarray)==i)

                x, y = indices[0][0], indices[1][0]
                sx, sy = sindices[0][0], sindices[1][0]

                self.assertEqual(p1[x,y], sp1[sx,sy])

    def test_player_strategy_extraction(self, how_many=100):
        for i in range(how_many):
            game = g.random_game().standard
            p0 = game.player(0).view(np.ndarray)
            p1 = game.player(1).view(np.ndarray)

            p0_direct = game[0].view(np.ndarray)
            p1_direct = game[1].view(np.ndarray)

            self.assertTrue((p0 == p0_direct).all())
            self.assertTrue((p1 == p1_direct).all())

    def test_row_and_column_sets_do_not_change(self, how_many=100):
        def get_row_column_sets(full_game):
            rows = sorted([sorted(
                zip(full_game[0, x].tolist(), full_game[1, x].tolist())
            ) for x in range(3)])
            columns = sorted([sorted(
                zip(full_game[0, :, x].tolist(), full_game[1, :, x])
            )
                           for x in range(3)])

            return tuple(rows), tuple(columns)

        for i in range(how_many):
            game = g.random_game()

            rc_before = get_row_column_sets(game)
            rc_after = get_row_column_sets(game.standard)

            self.assertEqual(rc_before, rc_after)


if __name__ == '__main__':
    half_tests = unittest.TestLoader().loadTestsFromTestCase(TestHalfGame)
    full_tests = unittest.TestLoader().loadTestsFromTestCase(TestGame)

    test_runner = unittest.TextTestRunner(verbosity=3)

    test_runner.run(half_tests)
    test_runner.run(full_tests)
