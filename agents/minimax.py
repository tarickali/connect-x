import math

import numpy as np

from connectx.types import State, Actions, Action, Config, Grid
import connectx.functional as cxf

__all__ = ["MinimaxAgent"]


class MinimaxAgent:
    def __init__(self, config: Config, depth: int = 3) -> None:
        self.config = config
        self.depth = depth

    def select(self, state: State, actions: Actions) -> Action:
        grid = state["grid"]
        info = state["info"]

        valid_cols = cxf.valid_action_columns(actions)
        if valid_cols.size == 0:
            # No valid actions; fall back to column 0 (should not happen in normal play).
            return np.uint8(0)

        players = self.config["players"]
        assert len(players) == 2, "MinimaxAgent currently supports exactly two players."

        active_index = info["active"]
        max_token = np.uint8(players[active_index])
        min_token = np.uint8(players[1 - active_index])

        best_score = float("-inf")
        best_col = int(valid_cols[0])

        for col in valid_cols:
            col_int = int(col)
            child_grid = cxf.place_token(grid, max_token, np.uint8(col_int))
            score = self.minimax(
                child_grid,
                depth=self.depth - 1,
                current_token=min_token,
                other_token=max_token,
                maximizing_token=max_token,
            )
            if score > best_score:
                best_score = score
                best_col = col_int

        return np.uint8(best_col)

    def minimax(
        self,
        grid: Grid,
        depth: int,
        current_token: int,
        other_token: int,
        maximizing_token: int,
    ) -> float:
        if cxf.terminal(grid, self.config["k"]):
            if not cxf.check_tie(grid):
                return 0.0

            if other_token == maximizing_token:
                return float("inf")
            else:
                return float("-inf")

        if depth == 0:
            return self.evaluate(grid, maximizing_token, other_token)

        actions = cxf.generate_actions(grid)
        valid_cols = cxf.valid_action_columns(actions)
        if valid_cols.size == 0:
            # No moves but not marked terminal: treat as neutral.
            return 0.0

        current_token_maximizing = current_token == maximizing_token
        best = float("-inf") if current_token_maximizing else float("inf")
        for col in valid_cols:
            col_int = int(col)
            child = cxf.place_token(grid, current_token, np.uint8(col_int))
            score = self.minimax(
                child,
                depth=depth - 1,
                current_token=other_token,
                other_token=current_token,
                maximizing_token=maximizing_token,
            )
            if (current_token_maximizing and score > best) or (
                not current_token_maximizing and score < best
            ):
                best = score
        return best

    def evaluate(
        self, grid: Grid, maximizing_token: int, minimizing_token: int
    ) -> float:
        score = 0.0
        rows, cols = grid.shape
        k = self.config["k"]
        max_token, min_token = int(maximizing_token), int(minimizing_token)

        # 1. Center column prefence
        # Rationale: middle columns are stronger
        center_col = cols // 2
        for c in range(cols):
            # Weight function: 1.0 + 2.0 * (distance % from center)
            weight = 1.0 + 2.0 * (1.0 - abs(c - center_col) / max(center_col, 1))
            for r in range(rows):
                if grid[r, c] == max_token:
                    score += weight
                elif grid[r, c] == min_token:
                    score -= weight

        # 2. Row height: prefer pieces lower on the board (more stable)
        for r in range(rows):
            for c in range(cols):
                if grid[r, c] == max_token:
                    score += 0.5 * r
                elif grid[r, c] == min_token:
                    score -= 0.5 * r

        # 3. Unblocked windows of length k (threat heuristic)
        # For each contiguous segment of length k, if it has no opponent piece it's a potential line
        score += self._score_windows(grid, k, max_token, min_token)

    ############################################################################
    # Helper
    ############################################################################
    def _score_windows(grid: Grid, k: int, max_token: int, min_token: int) -> float:
        def window_score(self, window: np.ndarray, max_t: int, min_t: int) -> float:
            n_max = int(np.sum(window == max_t))
            n_min = int(np.sum(window == min_t))
            if n_max > 0 and n_min > 0:
                return 0.0
            if n_max > 0:
                return 10.0**n_max
            if n_min > 0:
                return -(10.0**n_min)
            return 0.0

        total = 0.0
        rows, cols = grid.shape

        # Horizontal
        for r in range(rows):
            for c in range(cols - k + 1):
                window = grid[r, c : c + k]
                total += window_score(window, max_token, min_token)

        # Vertical
        for c in range(cols):
            for r in range(rows - k + 1):
                window = grid[r : r + k, c]
                total += window_score(window, max_token, min_token)

        # Main diagonal (top-left to bottom-right)
        for r in range(rows - k + 1):
            for c in range(cols - k + 1):
                window = np.array([grid[r + i, c + i] for i in range(k)])
                total += window_score(window, max_token, min_token)

        # Anti-diagonal (top-right to bottom-left)
        for r in range(rows - k + 1):
            for c in range(cols - k + 1):
                window = np.array([grid[r + i, c + k - 1 - i] for i in range(k)])
                total += window_score(window, max_token, min_token)

        return total
