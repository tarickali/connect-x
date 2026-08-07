"""Terminal rendering.

Kept separate from :mod:`connectx.game` so alternative front-ends (curses,
PyGame, notebook HTML) can be added without touching the engine.
"""

from collections.abc import Sequence

import numpy as np

import connectx.functional as cxf

__all__ = ["terminal_render", "grid_to_string"]

#: Symbols for tokens 1..n. Falls back to the raw number past the end.
_SYMBOLS = "XO*+#@%&"

_EMPTY = "."


def _symbol(token: int) -> str:
    if token == 0:
        return _EMPTY
    if token <= len(_SYMBOLS):
        return _SYMBOLS[token - 1]
    return str(token % 10)


def _winning_cells(grid: np.ndarray, k: int | None) -> set[tuple[int, int]]:
    """Cells on the winning line, so they can be marked in the output."""
    if k is None:
        return set()
    line = cxf.winner_line(grid, int(k))
    if line[0] < 0:
        return set()
    _, r0, c0, r1, c1 = (int(v) for v in line)
    steps = max(abs(r1 - r0), abs(c1 - c0))
    d_row = 0 if steps == 0 else (r1 - r0) // steps
    d_col = 0 if steps == 0 else (c1 - c0) // steps
    return {(r0 + i * d_row, c0 + i * d_col) for i in range(steps + 1)}


def grid_to_string(
    grid: np.ndarray,
    *,
    k: int | None = None,
    symbols: bool = True,
) -> str:
    """Render a grid as text, bracketing the winning line if there is one.

    Set ``symbols=False`` to print raw token numbers instead of X/O glyphs,
    which is easier to read for variants with many players.
    """
    rows, cols = grid.shape
    highlight = _winning_cells(grid, k)
    width = 1 if symbols else max(1, len(str(int(grid.max(initial=0)))))

    lines = ["  " + " ".join(f"{c:>{width}}" for c in range(cols))]
    lines.append(" +" + "-" * (cols * (width + 1) + 1) + "+")
    for row in range(rows):
        cells = []
        for col in range(cols):
            token = int(grid[row, col])
            text = _symbol(token) if symbols else str(token)
            cells.append(f"{text:>{width}}")
        body = " ".join(cells)
        if any((row, col) in highlight for col in range(cols)):
            # Mark the row containing the win so it is visible at a glance.
            lines.append(f" | {body} |*")
        else:
            lines.append(f" | {body} |")
    lines.append(" +" + "-" * (cols * (width + 1) + 1) + "+")
    return "\n".join(lines)


def terminal_render(
    grid: np.ndarray,
    time: int,
    player: int,
    *,
    k: int | None = None,
    symbols: bool = True,
    players: Sequence[int] | None = None,
) -> None:
    """Print the board with a header showing the turn and player to move."""
    header = f"time {time}  |  to move: {_symbol(player) if symbols else player}"
    if players is not None and symbols:
        legend = "  ".join(f"{_symbol(p)}={p}" for p in players)
        header = f"{header}  |  {legend}"
    print(header)
    print(grid_to_string(grid, k=k, symbols=symbols))
