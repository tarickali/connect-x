"""Pure, JIT-compiled primitives for Connect-style games.

A grid is a ``uint8`` array where ``0`` marks an empty cell and any non-zero
value is a player token. Every function here is a pure function of its inputs:
none of them mutate the grid they are given.

Two families of win checks are provided:

- :func:`winner` scans the whole board and is used when resuming from an
  arbitrary position.
- :func:`winner_at` only looks along the four lines through a single cell and
  is ``O(k)``. Use it after a move, when you know where the token landed.
"""

import numba as nb
import numpy as np

from connectx.types import Action, Actions, Grid, Shape

__all__ = [
    "EMPTY",
    "GRID_ARG",
    "MASK_ARG",
    "create_grid",
    "drop_row",
    "is_legal",
    "place_token",
    "generate_actions",
    "valid_action_columns",
    "full",
    "winner",
    "winner_at",
    "winner_line",
    "is_draw",
    "terminal",
]

EMPTY = np.uint8(0)

# (row, col) deltas for the four line orientations: -, |, \, /
_N_DIRECTIONS = 4

# Input array types are declared read-only so these functions accept the
# non-writeable views handed out by `Game.state` as well as ordinary arrays.
# Numba widens writable arrays to read-only automatically; the reverse is not
# allowed, which is exactly the guarantee we want.
GRID_ARG = nb.types.Array(nb.uint8, 2, "A", readonly=True)
MASK_ARG = nb.types.Array(nb.uint8, 1, "A", readonly=True)


@nb.njit(nb.types.UniTuple(nb.int64, 2)(nb.int64), cache=True)
def _direction(index: int) -> tuple[int, int]:
    """Return the (row, col) delta for one of the four line orientations."""
    if index == 0:
        return 0, 1
    if index == 1:
        return 1, 0
    if index == 2:
        return 1, 1
    return 1, -1


@nb.njit(nb.uint8[:, :](nb.types.UniTuple(nb.int64, 2)), cache=True)
def create_grid(shape: Shape) -> Grid:
    """Return an empty ``(rows, cols)`` grid."""
    return np.zeros(shape, dtype=np.uint8)


@nb.njit(nb.int64(GRID_ARG, nb.int64), cache=True)
def drop_row(grid: Grid, action: Action) -> int:
    """Return the row a token dropped into ``action`` would land on.

    Returns ``-1`` when the column is full or the index is out of range, which
    makes this the cheapest legality check available.
    """
    rows, cols = grid.shape
    if action < 0 or action >= cols:
        return -1
    for row in range(rows - 1, -1, -1):
        if grid[row, action] == 0:
            return row
    return -1


@nb.njit(nb.boolean(GRID_ARG, nb.int64), cache=True)
def is_legal(grid: Grid, action: Action) -> bool:
    """Return whether ``action`` is a playable column on ``grid``."""
    return drop_row(grid, action) >= 0


@nb.njit(nb.uint8[:, :](GRID_ARG, nb.uint8, nb.int64), cache=True)
def place_token(grid: Grid, token: np.uint8, action: Action) -> Grid:
    """Return a new grid with ``token`` dropped into column ``action``.

    Illegal actions return an unchanged copy. Callers that need to distinguish
    "played" from "ignored" should check :func:`is_legal` or :func:`drop_row`
    first; :class:`connectx.game.Game` does this and raises instead.
    """
    out = np.copy(grid)
    row = drop_row(grid, action)
    if row < 0:
        return out
    out[row, action] = token
    return out


@nb.njit(nb.uint8[:](GRID_ARG), cache=True)
def generate_actions(grid: Grid) -> Actions:
    """Return a ``uint8`` mask of playable columns (1 = legal)."""
    cols = grid.shape[1]
    actions = np.zeros(cols, np.uint8)
    for col in range(cols):
        if drop_row(grid, col) >= 0:
            actions[col] = 1
    return actions


@nb.njit(nb.int64[:](MASK_ARG), cache=True)
def valid_action_columns(actions: Actions) -> np.ndarray:
    """Convert an action mask into the array of legal column indices."""
    n = actions.shape[0]
    count = 0
    for i in range(n):
        if actions[i] == 1:
            count += 1
    out = np.empty(count, dtype=np.int64)
    j = 0
    for i in range(n):
        if actions[i] == 1:
            out[j] = i
            j += 1
    return out


@nb.njit(nb.boolean(GRID_ARG), cache=True)
def full(grid: Grid) -> bool:
    """Return whether every cell is occupied."""
    rows, cols = grid.shape
    for row in range(rows):
        for col in range(cols):
            if grid[row, col] == 0:
                return False
    return True


@nb.njit(nb.uint8(GRID_ARG, nb.int64, nb.int64, nb.int64), cache=True)
def winner_at(grid: Grid, k: int, row: int, col: int) -> np.uint8:
    """Return the token owning a length-``k`` line through ``(row, col)``, else 0.

    ``O(k)``: only the four lines through the cell are examined. Call this after
    a move with the cell the token landed on.
    """
    rows, cols = grid.shape
    if row < 0 or row >= rows or col < 0 or col >= cols:
        return EMPTY
    token = grid[row, col]
    if token == 0 or k <= 0:
        return EMPTY

    for index in range(_N_DIRECTIONS):
        d_row, d_col = _direction(index)
        count = 1

        r, c = row + d_row, col + d_col
        while 0 <= r and r < rows and 0 <= c and c < cols and grid[r, c] == token:
            count += 1
            r += d_row
            c += d_col

        r, c = row - d_row, col - d_col
        while 0 <= r and r < rows and 0 <= c and c < cols and grid[r, c] == token:
            count += 1
            r -= d_row
            c -= d_col

        if count >= k:
            return token
    return EMPTY


@nb.njit(nb.int64[:](GRID_ARG, nb.int64), cache=True)
def winner_line(grid: Grid, k: int) -> np.ndarray:
    """Return ``[token, row0, col0, row1, col1]`` for a winning line, or all -1.

    Scans the board once. Each run is counted from its first cell only, so the
    total work is ``O(rows * cols)`` rather than ``O(rows * cols * k)``.
    """
    out = np.full(5, -1, dtype=np.int64)
    rows, cols = grid.shape
    if k <= 0:
        return out

    for row in range(rows):
        for col in range(cols):
            token = grid[row, col]
            if token == 0:
                continue
            for index in range(_N_DIRECTIONS):
                d_row, d_col = _direction(index)

                # Skip cells that continue a run started earlier.
                p_row, p_col = row - d_row, col - d_col
                if (
                    0 <= p_row
                    and p_row < rows
                    and 0 <= p_col
                    and p_col < cols
                    and grid[p_row, p_col] == token
                ):
                    continue

                count = 0
                r, c = row, col
                while 0 <= r and r < rows and 0 <= c and c < cols and grid[r, c] == token:
                    count += 1
                    if count >= k:
                        out[0] = token
                        out[1] = row
                        out[2] = col
                        out[3] = r
                        out[4] = c
                        return out
                    r += d_row
                    c += d_col
    return out


@nb.njit(nb.uint8(GRID_ARG, nb.int64), cache=True)
def winner(grid: Grid, k: int) -> np.uint8:
    """Return the token with a length-``k`` line anywhere on ``grid``, else 0."""
    line = winner_line(grid, k)
    if line[0] < 0:
        return EMPTY
    return np.uint8(line[0])


@nb.njit(nb.boolean(GRID_ARG, nb.int64), cache=True)
def is_draw(grid: Grid, k: int) -> bool:
    """Return whether the board is full *and* nobody has a line."""
    return full(grid) and winner(grid, k) == 0


@nb.njit(nb.boolean(GRID_ARG, nb.int64), cache=True)
def terminal(grid: Grid, k: int) -> bool:
    """Return whether the game is over: someone has a line, or the board is full."""
    return winner(grid, k) != 0 or full(grid)
