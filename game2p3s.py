import numpy as np
import itertools as it
import logging
import copy

logger = logging.getLogger('Game2p3s')

# (players, strategies, strategies)
GAME_SHAPE = (2, 3, 3)


def pole_position_max(game_array):
    game_array = game_array.copy()

    for axis, roll_by in enumerate(np.unravel_index(game_array[0].argmax(), game_array.shape)[1:]):
        if roll_by:
            game_array = np.roll(game_array, -roll_by, axis=axis+1)

    return game_array


def random_game(as_nxngame=True, auto_canonicalize=True):
    players = [np.arange(3**2) + 1 for i in range(2)]
    for i, player in enumerate(players):
        np.random.shuffle(player)

    game_array = np.array(players).reshape((2, 3, 3))

    if as_nxngame:
        return Game2p3s(game_array, auto_canonicalize=auto_canonicalize)
    else:
        return game_array


class Game2p3s(object):
    def __init__(self, game_array=None, auto_canonicalize=True):
        if game_array is None:
            raise GameException("Game array must be provided.  Use random_game() to get a " +
                                "valid, random game.")



        self._game_array = np.array(game_array)

        if auto_canonicalize:
            self.canonical_form(write_back=True)

    def __str__(self):
        return "Game2p3s:\nP1:\n{}\nP2:\n{}".format(
            self._game_array[0, :,:].__str__(),
            self._game_array[1, :,:].__str__(),
        )

    def __repr__(self):
        return "{}({})".format(
            type(self).__name__,
            self._game_array.tolist(),
        )

    @property
    def shape(self):
        return self._game_array.shape

    def __getitem__(self, item):
        return self.game_array.__getitem__(item)

    @property
    def game_array(self):
        return self._game_array

    @game_array.setter
    def game_array(self, new_game_array):
        self._game_array = np.array(new_game_array).reshape((2, 3, 3))

    def canonical_form(self, write_back=False):
        game_array = self._game_array.copy()

        game_array = pole_position_max(game_array)
        game_array[:, 1:, 1:] = pole_position_max(game_array[:, 1:, 1:])

        if write_back:
            self._game_array = game_array

        return game_array

    def player(self, which_player):
        if which_player not in (0,1):
            raise GameException("Invalid player indexed.")

        return self._game_array[which_player]


class GameException(Exception): pass