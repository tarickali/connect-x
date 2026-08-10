"""Negamax search with alpha-beta pruning, move ordering, and a transposition table.

Written as negamax rather than an explicit max/min pair: the value is always
returned from the perspective of the side to move, so there is no pair of
``maximizing_token`` / ``minimizing_token`` arguments to get out of step with
each other. That pairing was the original bug.

The search walks the tree through :class:`agents.search.Position`, so it never
assumes how a move is applied and runs on any engine. Two-player only; negamax
assumes a zero-sum, strictly alternating game. Use
:class:`~agents.mcts.MCTSAgent` for variants with more players.
"""

from __future__ import annotations

import time

import numpy as np

from agents.heuristics import centre_out_order, evaluate
from agents.search import Position
from agents.types import BaseAgent
from connectx.engine import GameEngine
from connectx.types import Action, Actions, Config, State

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
        self._order: list[int] = []
        self._search: GameEngine | None = None
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
        self._search = self.engine_factory(config, undo=True)
        self._order = centre_out_order(self._search.action_space_size)
        self._table.clear()

    # ------------------------------------------------------------------

    def select(self, state: State, actions: Actions) -> Action:
        if self.config is None or self._search is None:
            raise RuntimeError(
                "MinimaxAgent needs a config; pass one to the constructor or call reset()."
            )
        position = Position(self._search, state)
        ordered = position.legal_ordered(self._order)
        if not ordered:
            raise ValueError("no legal actions available")
        if len(ordered) == 1:
            return ordered[0]

        self.nodes = 0
        if self.time_limit is None:
            self._deadline = None
            return self._search_root(position, ordered, self.depth)

        self._deadline = time.perf_counter() + float(self.time_limit)
        best = ordered[0]
        for depth in range(1, self.depth + 1):
            try:
                best = self._search_root(position, ordered, depth)
            except _Timeout:
                position.unwind()
                break
            # Put the best move first so the next iteration prunes harder.
            ordered.remove(best)
            ordered.insert(0, best)
        return best

    def _search_root(self, position: Position, actions: list[int], depth: int) -> Action:
        best_action = actions[0]
        best_value = -np.inf
        alpha = -np.inf
        seat = position.seat
        for action in actions:
            position.push(action)
            try:
                if position.winner_seat() == seat:
                    return action
                value = -self._negamax(position, depth - 1, -np.inf, -alpha, 1)
            finally:
                position.pop()
            if value > best_value:
                best_value = value
                best_action = action
            if value > alpha:
                alpha = value
        return best_action

    def _negamax(
        self, position: Position, depth: int, alpha: float, beta: float, ply: int
    ) -> float:
        """Value of the current position for the side to move."""
        self.nodes += 1
        if self._deadline is not None and self.nodes % 2048 == 0:
            if time.perf_counter() >= self._deadline:
                raise _Timeout

        if position.terminal():
            # Whoever just moved either won or drew; the side to move never wins
            # on the opponent's move, so a winner here means we lost.
            return 0.0 if position.winner_seat() is None else -(WIN_SCORE - ply)
        if depth <= 0:
            return self._evaluate(position)

        key = (
            (position.grid.tobytes(), position.seat) if self.use_transpositions else None
        )
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
        for action in position.legal_ordered(self._order):
            position.push(action)
            try:
                value = -self._negamax(position, depth - 1, -beta, -alpha, ply + 1)
            finally:
                position.pop()
            if value > best:
                best = value
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break

        if best == -np.inf:  # no legal move: the position is a dead end
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

    def _evaluate(self, position: Position) -> float:
        seat = position.seat
        own = np.uint8(position.players[seat])
        opponent = np.uint8(position.players[position.opponent_of(seat)])
        return float(evaluate(position.grid, self._k, own, opponent))
