"""UCT Monte Carlo tree search.

Backs up a reward *vector* rather than a scalar, and each node selects using
the entry belonging to the player to move at that node. That is the max^n
generalization of UCT, so this agent works unchanged on three- and four-player
variants where negamax does not apply — which matters, because otherwise the
multi-player configs would have no strong baseline to measure against.

The tree itself is plain Python, and profiling showed that is where the time
goes: only about a quarter of a search is spent inside the compiled rollout.
So the hot paths here avoid per-simulation allocation — reward vectors are
precomputed per winning token in :meth:`reset`, node statistics are Python
floats rather than one-element-at-a-time numpy arrays, and expansion reuses a
single legality mask instead of rebuilding index arrays.
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
        # A Python list, not an ndarray: these are updated once per node per
        # simulation, where numpy's per-call overhead dwarfs the arithmetic.
        self.values: list[float] = [0.0] * n_players
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
        self._order: list[int] = []
        self._players: np.ndarray = np.zeros(0, dtype=np.uint8)
        self._n_players = 0
        self._payoffs: dict[int, list[float]] = {}
        super().__init__(config, **kwargs)

    def reset(self, config: Config) -> None:
        super().reset(config)
        self._k = int(config["k"])
        # Held as a Python list; it is walked once per expansion and per
        # selection, and numpy scalar unboxing is not free at that rate.
        self._order = [int(c) for c in column_order(int(config["shape"][1]))]
        self._players = np.array(config["players"], dtype=np.uint8)
        self._n_players = len(config["players"])

        # Precompute the backup vector for every outcome, so a simulation never
        # allocates. Losses share the win evenly, keeping the game zero-sum for
        # any number of seats.
        loss = -1.0 / max(self._n_players - 1, 1)
        self._payoffs = {0: [0.0] * self._n_players}
        for seat, token in enumerate(config["players"]):
            payoff = [loss] * self._n_players
            payoff[seat] = 1.0
            self._payoffs[int(token)] = payoff

    # ------------------------------------------------------------------

    def select(self, state: State, actions: Actions) -> Action:
        if self.config is None:
            raise RuntimeError(
                "MCTSAgent needs a config; pass one to the constructor or call reset()."
            )
        grid = np.ascontiguousarray(state["grid"], dtype=np.uint8)
        legal = [column for column in self._order if actions[column] == 1]
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
            untried=legal,
            n_players=self._n_players,
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
            self._backpropagate(node, self._simulate(node))
            simulation += 1

        self.last_visits = {a: c.visits for a, c in root.children.items()}
        if not root.children:
            return legal[0]
        # Most-visited is the robust choice; it is less sensitive to a single
        # lucky rollout than picking the highest mean value.
        return max(root.children.items(), key=lambda kv: kv[1].visits)[0]

    # ------------------------------------------------------------------

    def _legal_ordered(self, grid: Grid) -> list[int]:
        """Centre-out legal columns, from one mask rather than three arrays."""
        mask = cxf.generate_actions(grid)
        return [column for column in self._order if mask[column] == 1]

    def _select_leaf(self, node: _Node) -> _Node:
        while not node.is_terminal and not node.untried and node.children:
            node = self._best_child(node)
        return node

    def _best_child(self, node: _Node) -> _Node:
        exploration = self.exploration
        log_parent = math.log(node.visits) if node.visits > 0 else 0.0
        seat = node.seat
        best_score = -math.inf
        best = node
        for child in node.children.values():
            visits = child.visits
            if visits == 0:
                return child
            score = child.values[seat] / visits + exploration * math.sqrt(
                log_parent / visits
            )
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
        child = _Node(
            grid=grid,
            seat=(node.seat + 1) % self._n_players,
            parent=node,
            action=column,
            untried=[] if is_terminal else self._legal_ordered(grid),
            n_players=self._n_players,
            terminal_token=winner_token,
            is_terminal=is_terminal,
        )
        node.children[column] = child
        return child

    def _simulate(self, node: _Node) -> list[float]:
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
        return self._payoffs[token]

    def _backpropagate(self, node: _Node | None, payoff: list[float]) -> None:
        seats = range(self._n_players)
        while node is not None:
            node.visits += 1
            values = node.values
            for seat in seats:
                values[seat] += payoff[seat]
            node = node.parent
