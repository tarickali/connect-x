"""Negamax search with alpha-beta pruning, move ordering, and a transposition table.

Written as negamax rather than an explicit max/min pair: the value is always
returned from the perspective of the side to move, so there is no pair of
``maximizing_token`` / ``minimizing_token`` arguments to get out of step with
each other. Terminal detection uses :func:`connectx.functional.winner_at` on
the cell the last token landed on, which is ``O(k)`` instead of a full rescan.

Two-player only; negamax assumes a zero-sum, strictly alternating game. Use
:class:`~agents.mcts.MCTSAgent` for variants with more players.
"""

from __future__ import annotations

import time

import numpy as np

import connectx.functional as cxf
from agents.heuristics import column_order, evaluate
from agents.types import BaseAgent
from connectx.types import Action, Actions, Config, Grid, State

__all__ = ["MinimaxAgent"]

#: Score of an immediate win. Wins are discounted by ply so the search prefers
#: to win sooner and to lose later, instead of treating all wins as equal.
WIN_SCORE = 1e9

_EXACT, _LOWER, _UPPER = 0, 1, 2

#: Above this magnitude a score is a mate distance, which is only valid at the
#: ply it was found; those are never cached.
_MATE_THRESHOLD = WIN_SCORE / 2

_MAX_TT_ENTRIES = 1 << 20


class _Timeout(Exception):
    """Raised internally to abandon an iterative-deepening iteration."""


class MinimaxAgent(BaseAgent):
    """Alpha-beta search to a fixed depth, or to whatever depth fits a time budget.

    Set ``time_limit`` (seconds) to search by iterative deepening instead of a
    fixed ``depth``; the best move from the last completed iteration is kept.
    That makes equal-time matchups possible in the arena, which is a fairer
    comparison than equal depth when board sizes differ.
    """

    def __init__(
        self,
        config: Config | None = None,
        depth: int = 4,
        *,
        time_limit: float | None = None,
        use_transpositions: bool = True,
        **kwargs,
    ) -> None:
        self.depth = int(depth)
        self.time_limit = time_limit
        self.use_transpositions = use_transpositions
        self.nodes = 0
        self._k = 0
        self._order: np.ndarray = np.zeros(0, dtype=np.int64)
        self._tokens: tuple[int, int] = (1, 2)
        self._table: dict[tuple[bytes, int], tuple[int, int, float]] = {}
        self._deadline: float | None = None
        super().__init__(config, **kwargs)

    # ------------------------------------------------------------------

    def reset(self, config: Config) -> None:
        players = config["players"]
        if len(players) != 2:
            raise ValueError(
                f"MinimaxAgent supports exactly two players, got {len(players)}. "
                "Use MCTSAgent for variants with more."
            )
        super().reset(config)
        self._k = int(config["k"])
        self._order = column_order(int(config["shape"][1]))
        self._tokens = (int(players[0]), int(players[1]))
        self._table.clear()

    # ------------------------------------------------------------------

    def select(self, state: State, actions: Actions) -> Action:
        if self.config is None:
            raise RuntimeError(
                "MinimaxAgent needs a config; pass one to the constructor or call reset()."
            )
        grid = np.ascontiguousarray(state["grid"], dtype=np.uint8)
        columns = cxf.valid_action_columns(actions)
        if columns.size == 0:
            raise ValueError("no legal actions available")

        me = np.uint8(self.config["players"][state["info"]["active"]])
        other = self._tokens[0] if int(me) == self._tokens[1] else self._tokens[1]
        opponent = np.uint8(other)

        legal = {int(c) for c in columns}
        ordered = [int(c) for c in self._order if int(c) in legal]
        if len(ordered) == 1:
            return ordered[0]

        self.nodes = 0
        if self.time_limit is None:
            self._deadline = None
            return self._search_root(grid, ordered, me, opponent, self.depth)

        self._deadline = time.perf_counter() + float(self.time_limit)
        best = ordered[0]
        for depth in range(1, self.depth + 1):
            try:
                best = self._search_root(grid, ordered, me, opponent, depth)
            except _Timeout:
                break
            # Put the best move first so the next iteration prunes harder.
            ordered.remove(best)
            ordered.insert(0, best)
        return best

    def _search_root(
        self,
        grid: Grid,
        columns: list[int],
        me: np.uint8,
        opponent: np.uint8,
        depth: int,
    ) -> Action:
        best_column = columns[0]
        best_value = -np.inf
        alpha = -np.inf
        for column in columns:
            row = cxf.drop_row(grid, column)
            child = cxf.place_token(grid, me, column)
            if cxf.winner_at(child, self._k, row, column) == me:
                return column
            value = -self._negamax(
                child, depth - 1, -np.inf, -alpha, opponent, me, row, column, 1
            )
            if value > best_value:
                best_value = value
                best_column = column
            if value > alpha:
                alpha = value
        return best_column

    def _negamax(
        self,
        grid: Grid,
        depth: int,
        alpha: float,
        beta: float,
        token: np.uint8,
        opponent: np.uint8,
        last_row: int,
        last_column: int,
        ply: int,
    ) -> float:
        """Value of ``grid`` for ``token`` to move, given ``opponent`` just played."""
        self.nodes += 1
        if self._deadline is not None and self.nodes % 2048 == 0:
            if time.perf_counter() >= self._deadline:
                raise _Timeout

        # The opponent's move may have ended the game; the side to move lost.
        if cxf.winner_at(grid, self._k, last_row, last_column) != 0:
            return -(WIN_SCORE - ply)
        if cxf.full(grid):
            return 0.0
        if depth <= 0:
            return float(evaluate(grid, self._k, token, opponent))

        key = (grid.tobytes(), int(token)) if self.use_transpositions else None
        alpha_original = alpha
        if key is not None:
            cached = self._table.get(key)
            if cached is not None and cached[0] >= depth:
                _, flag, value = cached
                if flag == _EXACT:
                    return value
                if flag == _LOWER:
                    alpha = max(alpha, value)
                elif flag == _UPPER:
                    beta = min(beta, value)
                if alpha >= beta:
                    return value

        best = -np.inf
        for column in self._order:
            column = int(column)
            row = cxf.drop_row(grid, column)
            if row < 0:
                continue
            child = cxf.place_token(grid, token, column)
            value = -self._negamax(
                child, depth - 1, -beta, -alpha, opponent, token, row, column, ply + 1
            )
            if value > best:
                best = value
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break

        if best == -np.inf:  # no legal move: board filled up
            return 0.0

        if key is not None and abs(best) < _MATE_THRESHOLD:
            if len(self._table) >= _MAX_TT_ENTRIES:
                self._table.clear()
            if best <= alpha_original:
                flag = _UPPER
            elif best >= beta:
                flag = _LOWER
            else:
                flag = _EXACT
            self._table[key] = (depth, flag, best)
        return best
