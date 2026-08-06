import numpy as np
import numba as nb

from connectx.types import Shape, Grid, Action, Actions


@nb.njit(nb.uint8[:, :](nb.types.UniTuple(nb.uint8, 2)))
def create_grid(shape: Shape) -> Grid:
    return np.zeros(shape, dtype=np.uint8)


@nb.njit(nb.uint8[:, :](nb.uint8[:, :], nb.uint8, nb.uint8))
def place_token(grid: Grid, token: np.uint8, action: Action) -> Grid:
    out = np.copy(grid)
    # Top row occupied => column is full; avoid negative indexing into row -1.
    if out.shape[0] > 0 and out[0, action] != 0:
        return out
    for row in range(out.shape[0]):
        if out[row][action] != 0:
            out[row - 1][action] = token
            break
    else:
        out[out.shape[0] - 1][action] = token
    return out


@nb.njit(nb.uint8[:](nb.uint8[:, :]))
def generate_actions(grid: Grid) -> Actions:
    actions = np.zeros(grid.shape[1], np.uint8)
    for col in range(grid.shape[1]):
        if not np.all(grid[:, col] != 0):
            actions[col] = 1
    return actions


@nb.njit(nb.int64[:](nb.uint8[:]))
def valid_action_columns(actions: Actions) -> np.ndarray:
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


@nb.njit(nb.boolean(nb.uint8[:]))
def check_line(line: np.ndarray) -> bool:
    return line[0] != 0 and np.min(line) == np.max(line)


@nb.njit(nb.boolean(nb.uint8[:, :], nb.uint8))
def check_horizontal_lines(
    grid: Grid, k: np.uint8
) -> bool:  # tuple[bool, tuple[tuple[int, int], tuple[int, int]]]:
    for row in range(grid.shape[0]):
        # Get the columns of the row
        subgrid = grid[row, :]
        assert subgrid.ndim == 1 and subgrid.shape[0] == grid.shape[1]
        # Loop for all valid length k lines of row
        for col in range(grid.shape[1] - k + 1):
            # Get the length k line starting from k of row
            line = subgrid[col : col + k]
            assert line.shape[0] == k
            # Check if line is full
            if check_line(line):
                return True
    return False


@nb.njit(nb.boolean(nb.uint8[:, :], nb.uint8))
def check_vertical_lines(
    grid: Grid, k: np.uint8
) -> bool:  # tuple[bool, tuple[tuple[int, int], tuple[int, int]]]:
    for col in range(grid.shape[1]):
        # Get the rows of the col
        subgrid = grid[:, col]
        assert subgrid.ndim == 1 and subgrid.shape[0] == grid.shape[0]
        # Loop for all valid length k lines of col
        for row in range(grid.shape[0] - k + 1):
            # Get the length k line starting from k of col
            line = subgrid[row : row + k]
            assert line.shape[0] == k
            # Check if line is full
            if check_line(line):
                return True
    return False


@nb.njit(nb.boolean(nb.uint8[:, :], nb.uint8))
def check_diagonal_lines(grid: Grid, k: np.uint8) -> bool:
    for row in range(grid.shape[0]):
        # Check if row + k goes out of bounds of the grid
        if row + k > grid.shape[0]:
            break
        for col in range(grid.shape[1]):
            # Check if col + k goes out of bounds of the grid
            if col + k > grid.shape[1]:
                break
            # Get the k x k subgrid to check the diagonals
            subgrid = grid[row : row + k, col : col + k]
            assert subgrid.shape[0] == subgrid.shape[1] == k
            # Get the main and anti diagonals of the subgrid
            lr_line = np.diag(subgrid)
            rl_line = np.diag(np.fliplr(subgrid))
            # Check if either of the lines are full
            if check_line(lr_line) or check_line(rl_line):
                return True
    return False


@nb.njit(nb.boolean(nb.uint8[:, :]))
def check_tie(grid: Grid) -> bool:
    return np.all(grid != 0)


@nb.njit(nb.boolean(nb.uint8[:, :], nb.uint8))
def terminal(grid: Grid, k: int) -> bool:
    return (
        check_horizontal_lines(grid, k)
        or check_vertical_lines(grid, k)
        or check_diagonal_lines(grid, k)
        or check_tie(grid)
    )
