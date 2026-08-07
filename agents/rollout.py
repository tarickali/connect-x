"""JIT-compiled playouts for Monte Carlo search.

A rollout is the hot loop of MCTS, so it runs entirely inside ``numba``: the
board is copied once and then mutated in place, column heights are tracked
incrementally, and the win test only looks at the cell just filled. That avoids
the per-move array allocation the copy-on-write functional API would otherwise
incur thousands of times per move.
"""

import numba as nb
import numpy as np

from connectx.functional import GRID_ARG, MASK_ARG, winner_at
from connectx.types import Grid

__all__ = ["seed_rng", "playout"]


@nb.njit(nb.void(nb.int64), cache=True)
def seed_rng(value: int) -> None:
    """Seed numba's internal RNG so playouts are reproducible."""
    np.random.seed(value)


@nb.njit(nb.uint8(GRID_ARG, nb.int64, MASK_ARG, nb.int64, nb.boolean), cache=True)
def playout(
    grid: Grid, k: int, players: np.ndarray, active: int, greedy: bool
) -> np.uint8:
    """Play to the end from ``grid``; return the winning token, or 0 for a draw.

    With ``greedy=True`` a player takes an immediate win when one is on offer,
    which makes the value estimate far less noisy for a handful of extra
    ``winner_at`` calls per move.
    """
    board = np.copy(grid)
    rows, cols = board.shape
    n_players = players.shape[0]

    heights = np.zeros(cols, dtype=np.int64)
    for col in range(cols):
        filled = 0
        for row in range(rows):
            if board[row, col] != 0:
                filled += 1
        heights[col] = filled

    open_columns = np.empty(cols, dtype=np.int64)
    seat = active

    while True:
        n_open = 0
        for col in range(cols):
            if heights[col] < rows:
                open_columns[n_open] = col
                n_open += 1
        if n_open == 0:
            return np.uint8(0)

        token = players[seat]
        choice = -1

        if greedy:
            for i in range(n_open):
                col = open_columns[i]
                row = rows - 1 - heights[col]
                board[row, col] = token
                won = winner_at(board, k, row, col)
                board[row, col] = 0
                if won != 0:
                    choice = col
                    break

        if choice < 0:
            choice = open_columns[np.random.randint(n_open)]

        row = rows - 1 - heights[choice]
        board[row, choice] = token
        heights[choice] += 1
        if winner_at(board, k, row, choice) != 0:
            return token
        seat = (seat + 1) % n_players
