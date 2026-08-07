"""UCT Monte Carlo tree search.

Backs up a reward *vector* rather than a scalar, and each node selects using
the entry belonging to the player to move at that node. That is the max^n
generalization of UCT, so this agent works unchanged on three- and four-player
variants where negamax does not apply — which matters, because otherwise the
multi-player configs would have no strong baseline to measure against.
"""

from __future__ import annotations

import math
import time

import numpy as np

import connectx.functional as cxf
from agents.heuristics import column_order
from agents.rollout import playout, seed_rng
from agents.types import BaseAgent
from connectx.types import Action, Actions, Config, Grid, State

__all__ = ["MCTSAgent"]


class _Node:
    __slots__ = (
        "grid",
        "seat",
        "parent",
        "action",
        "children",
        "untried",
        "visits",
        "values",
        "terminal_token",
        "is_terminal",
    )

    def __init__(
        self,
        grid: Grid,
        seat: int,
        parent: _Node | None,
        action: int | None,
        untried: list[int],
        n_players: int,
        terminal_token: int,
        is_terminal: bool,
    ) -> None:
        self.grid = grid
        self.seat = seat
        self.parent = parent
        self.action = action
        self.children: dict[int, _Node] = {}
        self.untried = untried
        self.visits = 0
        self.values = np.zeros(n_players, dtype=np.float64)
        self.terminal_token = terminal_token
        self.is_terminal = is_terminal


class MCTSAgent(BaseAgent):
    """Search by simulation. Budget with ``simulations`` or ``time_limit``."""

    def __init__(
        self,
        config: Config | None = None,
        *,
        simulations: int = 400,
        time_limit: float | None = None,
        exploration: float = 1.4,
        greedy_rollouts: bool = True,
        **kwargs,
    ) -> None:
        self.simulations = int(simulations)
        self.time_limit = time_limit
        self.exploration = float(exploration)
        self.greedy_rollouts = bool(greedy_rollouts)
        self.last_visits: dict[int, int] = {}
        self._k = 0
        self._order: np.ndarray = np.zeros(0, dtype=np.int64)
        self._players: np.ndarray = np.zeros(0, dtype=np.uint8)
        super().__init__(config, **kwargs)

    def reset(self, config: Config) -> None:
        super().reset(config)
        self._k = int(config["k"])
        self._order = column_order(int(config["shape"][1]))
        self._players = np.array(config["players"], dtype=np.uint8)

    # ------------------------------------------------------------------

    def select(self, state: State, actions: Actions) -> Action:
        if self.config is None:
            raise RuntimeError(
                "MCTSAgent needs a config; pass one to the constructor or call reset()."
            )
        grid = np.ascontiguousarray(state["grid"], dtype=np.uint8)
        legal = [int(c) for c in cxf.valid_action_columns(actions)]
        if not legal:
            raise ValueError("no legal actions available")
        if len(legal) == 1:
            self.last_visits = {legal[0]: 0}
            return legal[0]

        # Drive numba's RNG from this agent's generator, keeping runs reproducible.
        seed_rng(int(self.rng.integers(1, 2**31 - 1)))

        seat = int(state["info"]["active"])
        root = _Node(
            grid=grid,
            seat=seat,
            parent=None,
            action=None,
            untried=self._ordered(legal),
            n_players=self._players.shape[0],
            terminal_token=0,
            is_terminal=False,
        )

        deadline = (
            None if self.time_limit is None else time.perf_counter() + self.time_limit
        )
        simulation = 0
        while simulation < self.simulations:
            if deadline is not None and time.perf_counter() >= deadline:
                break
            node = self._select_leaf(root)
            if not node.is_terminal and node.untried:
                node = self._expand(node)
            rewards = self._simulate(node)
            self._backpropagate(node, rewards)
            simulation += 1

        self.last_visits = {a: c.visits for a, c in root.children.items()}
        if not root.children:
            return legal[0]
        # Most-visited is the robust choice; it is less sensitive to a single
        # lucky rollout than picking the highest mean value.
        return max(root.children.items(), key=lambda kv: kv[1].visits)[0]

    # ------------------------------------------------------------------

    def _ordered(self, legal: list[int]) -> list[int]:
        allowed = set(legal)
        return [int(c) for c in self._order if int(c) in allowed]

    def _select_leaf(self, node: _Node) -> _Node:
        while not node.is_terminal and not node.untried and node.children:
            node = self._best_child(node)
        return node

    def _best_child(self, node: _Node) -> _Node:
        log_parent = math.log(max(node.visits, 1))
        best_score = -math.inf
        best = next(iter(node.children.values()))
        for child in node.children.values():
            if child.visits == 0:
                return child
            exploit = child.values[node.seat] / child.visits
            explore = self.exploration * math.sqrt(log_parent / child.visits)
            score = exploit + explore
            if score > best_score:
                best_score = score
                best = child
        return best

    def _expand(self, node: _Node) -> _Node:
        column = node.untried.pop(0)
        token = self._players[node.seat]
        row = cxf.drop_row(node.grid, column)
        grid = cxf.place_token(node.grid, token, column)
        winner_token = int(cxf.winner_at(grid, self._k, row, column))
        is_terminal = winner_token != 0 or cxf.full(grid)
        next_seat = (node.seat + 1) % self._players.shape[0]
        child = _Node(
            grid=grid,
            seat=next_seat,
            parent=node,
            action=column,
            untried=(
                []
                if is_terminal
                else self._ordered(
                    [int(c) for c in cxf.valid_action_columns(cxf.generate_actions(grid))]
                )
            ),
            n_players=self._players.shape[0],
            terminal_token=winner_token,
            is_terminal=is_terminal,
        )
        node.children[column] = child
        return child

    def _simulate(self, node: _Node) -> np.ndarray:
        if node.is_terminal:
            token = node.terminal_token
        else:
            token = int(
                playout(
                    node.grid,
                    self._k,
                    self._players,
                    node.seat,
                    self.greedy_rollouts,
                )
            )
        return self._rewards_for(token)

    def _rewards_for(self, token: int) -> np.ndarray:
        n_players = self._players.shape[0]
        rewards = np.zeros(n_players, dtype=np.float64)
        if token == 0:
            return rewards
        winner_seat = int(np.flatnonzero(self._players == np.uint8(token))[0])
        rewards[:] = -1.0 / max(n_players - 1, 1)
        rewards[winner_seat] = 1.0
        return rewards

    def _backpropagate(self, node: _Node | None, rewards: np.ndarray) -> None:
        while node is not None:
            node.visits += 1
            node.values += rewards
            node = node.parent
