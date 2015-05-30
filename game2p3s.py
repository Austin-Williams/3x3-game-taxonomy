import numpy as np
import itertools as it
import logging
import copy

logger = logging.getLogger('game2p3s')


def random_half_game(as_numpy_array=False):
    numpy_half_array = np.arange(3**2) + 1
    np.random.shuffle(numpy_half_array)
    numpy_half_array = numpy_half_array.reshape((3, 3))

    if as_numpy_array:
        return numpy_half_array

    return HalfGameArray(numpy_half_array)


def random_game(as_numpy_array=False):
    numpy_array = np.array(
        [random_half_game(as_numpy_array=True) for i in range(2)]
    )

    if as_numpy_array:
        return numpy_array

    return GameArray(numpy_array)


class HalfGameArray(np.ndarray):
    """
    HalfGameArray provides an interface to a single player's strategy in a 2-player,
    3-strategy game. It is a subclass of numpy.ndarray, so all numpy things should operate correctly
    on it.

    Properties:
        standard: the half game in standard form

    A few illuminating doctests:

    >>> hga = HalfGameArray(range(1,10))
    >>> print hga
    [[1 2 3]
     [4 5 6]
     [7 8 9]]
    >>> print hga.standard
    [[9 7 8]
     [3 5 4]
     [6 2 1]]
    """

    def __new__(cls, input_array, meta=None):
        obj = np.asarray(input_array).reshape(3,3).view(cls)
        obj.meta = meta
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.meta = getattr(obj, 'meta', dict())

    def __eq__(self, other):
        return (self.standard.view(np.ndarray) == other.standard.view(np.ndarray)).all()

    @property
    def standard(self):
        game_array = self.copy()

        array_rolls = np.unravel_index(
            game_array.argmax(),
            game_array.shape
        )
        for axis, roll_by in enumerate(array_rolls):
            if roll_by:
                game_array = np.roll(game_array, -roll_by, axis=axis)

        subarray_rolls = np.unravel_index(
                    game_array[1:, 1:].argmax(),
                    game_array[1:, 1:].shape
                )
        for axis, roll_by in enumerate(subarray_rolls):
            if roll_by:
                game_array[1:, 1:] = np.roll(game_array[1:, 1:], -roll_by, axis=axis)

        return game_array


class GameArray(np.ndarray):
    """
    GameArray provides an interface to a 2-player, 3-strategy game.
    """

    def __new__(cls, data, meta=None):
        obj = np.asarray(data).reshape(2,3,3).view(cls)
        obj.meta = meta
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.meta = getattr(obj, 'meta', dict())

    def __eq__(self, other):
        return (self.standard.view(np.ndarray) == other.standard.view(np.ndarray)).all()

    def player(self, player):
        if player not in range(2):
            raise GameIndexError("Only players 0 and 1 exist.  Player {} does not.".format(player))
        return self[player].view(HalfGameArray)

    @property
    def standard(self):
        game_array = self.copy()

        array_rolls = np.unravel_index(
            game_array[0, :, :].argmax(),
            game_array.shape
        )[1:]
        for axis, roll_by in enumerate(array_rolls):
            if roll_by:
                game_array = np.roll(game_array, -roll_by, axis=axis+1)

        subarray_rolls = np.unravel_index(
                    game_array[0, 1:, 1:].argmax(),
                    game_array[0, 1:, 1:].shape
                )
        for axis, roll_by in enumerate(subarray_rolls):
            if roll_by:
                game_array[:, 1:, 1:] = np.roll(game_array[:, 1:, 1:], -roll_by, axis=axis+1)

        return game_array.view(self.__class__)


class GameBaseException(Exception): pass


class GameIndexError(GameBaseException): pass