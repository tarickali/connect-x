"""JIT-compiled board heuristics shared by the search agents.

The scoring is deliberately scale-free in ``k``: window value grows
quadratically with the number of own tokens rather than exponentially, so the
same evaluation works for ``k=3`` on a 4x4 board and ``k=6`` on a 20x20 one
without overflowing or swamping the search. That matters here — the point of
the project is that one agent should be able to play every variant.
"""

import numba as nb
import numpy as np

from connectx.functional import GRID_ARG
from connectx.types import Grid

__all__ = ["window_score", "center_score", "evaluate", "column_order"]

#: Extra credit for a window that is one token away from completing.
_THREAT_BONUS = 40.0

#: Opponent threats are weighted slightly above own, which prefers blocking to
#: building when the two are otherwise equal.
_DEFENSE = 1.15


@nb.njit(nb.types.UniTuple(nb.int64, 2)(nb.int64), cache=True)
def _direction(index: int) -> tuple[int, int]:
    if index == 0:
        return 0, 1
    if index == 1:
        return 1, 0
    if index == 2:
        return 1, 1
    return 1, -1


@nb.njit(nb.float64(GRID_ARG, nb.int64, nb.uint8, nb.uint8), cache=True)
def window_score(grid: Grid, k: int, own: np.uint8, opponent: np.uint8) -> float:
    """Score every length-``k`` window that is still winnable by exactly one side."""
    rows, cols = grid.shape
    total = 0.0
    for index in range(4):
        d_row, d_col = _direction(index)
        for row in range(rows):
            for col in range(cols):
                end_row = row + (k - 1) * d_row
                end_col = col + (k - 1) * d_col
                if end_row < 0 or end_row >= rows or end_col < 0 or end_col >= cols:
                    continue
                n_own = 0
                n_opponent = 0
                for i in range(k):
                    value = grid[row + i * d_row, col + i * d_col]
                    if value == own:
                        n_own += 1
                    elif value == opponent:
                        n_opponent += 1
                # A contested window can no longer become a line for either side.
                if n_own > 0 and n_opponent > 0:
                    continue
                if n_own > 0:
                    total += float(n_own * n_own)
                    if n_own == k - 1:
                        total += _THREAT_BONUS
                elif n_opponent > 0:
                    total -= _DEFENSE * float(n_opponent * n_opponent)
                    if n_opponent == k - 1:
                        total -= _DEFENSE * _THREAT_BONUS
    return total


@nb.njit(nb.float64(GRID_ARG, nb.uint8, nb.uint8), cache=True)
def center_score(grid: Grid, own: np.uint8, opponent: np.uint8) -> float:
    """Reward central columns, which sit in the most potential lines."""
    rows, cols = grid.shape
    center = (cols - 1) / 2.0
    span = max(center, 1.0)
    total = 0.0
    for col in range(cols):
        weight = 1.0 - abs(col - center) / span
        for row in range(rows):
            value = grid[row, col]
            if value == own:
                total += weight
            elif value == opponent:
                total -= weight
    return total


@nb.njit(nb.float64(GRID_ARG, nb.int64, nb.uint8, nb.uint8), cache=True)
def evaluate(grid: Grid, k: int, own: np.uint8, opponent: np.uint8) -> float:
    """Static evaluation of a non-terminal position, from ``own``'s perspective."""
    return window_score(grid, k, own, opponent) + 2.0 * center_score(grid, own, opponent)


def column_order(cols: int) -> np.ndarray:
    """Centre-out column ordering.

    Searching the middle first makes alpha-beta cut off far earlier, because
    good moves cluster there.
    """
    center = (cols - 1) / 2.0
    return np.array(
        sorted(range(cols), key=lambda c: (abs(c - center), c)), dtype=np.int64
    )
