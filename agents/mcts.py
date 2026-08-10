"""UCT Monte Carlo tree search.

Backs up a reward *vector* rather than a scalar, and each node selects using
the entry belonging to the player to move at that node. That is the max^n
generalization of UCT, so this agent works unchanged on three- and four-player
variants where negamax does not apply — which matters, because otherwise the
multi-player configs would have no strong baseline to measure against.

The tree is walked with a single :class:`agents.search.Position` cursor: the
search pushes moves on the way down and pops them on the way back, so nodes
store an action rather than a board and the agent never assumes how a move is
applied. Playouts use the engine's compiled ``rollout`` when it offers one and
step the engine otherwise.

The tree itself is plain Python, and profiling showed that is where the time
goes, so the hot paths avoid per-simulation allocation: reward vectors are
precomputed per outcome in :meth:`reset` and node statistics are Python floats.
"""

from __future__ import annotations

import math
import time

from agents.heuristics import centre_out_order
from agents.search import Position
from agents.types import BaseAgent
from connectx.engine import GameEngine
from connectx.types import Action, Actions, Config, State

__all__ = ["MCTSAgent"]


class _Node:
    __slots__ = (
        "seat",
        "parent",
        "action",
        "children",
        "untried",
        "visits",
        "values",
        "winner_seat",
        "is_terminal",
    )

    def __init__(
        self,
        seat: int,
        parent: _Node | None,
        action: int | None,
        untried: list[int],
        n_players: int,
        winner_seat: int,
        is_terminal: bool,
    ) -> None:
        self.seat = seat
        self.parent = parent
        self.action = action
        self.children: dict[int, _Node] = {}
        self.untried = untried
        self.visits = 0
        # A Python list, not an ndarray: these are updated once per node per
        # simulation, where numpy's per-call overhead dwarfs the arithmetic.
        self.values: list[float] = [0.0] * n_players
        self.winner_seat = winner_seat  # -1 when nobody has won
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
        self._players: list[int] = []
        self._n_players = 0
        self._payoffs: dict[int, list[float]] = {}
        self._search: GameEngine | None = None
        super().__init__(config, **kwargs)

    def reset(self, config: Config) -> None:
        super().reset(config)
        self._k = int(config["k"])
        self._search = self.engine_factory(config, undo=True)
        self._order = centre_out_order(self._search.action_space_size)
        self._players = [int(p) for p in config["players"]]
        self._n_players = len(self._players)

        # Precompute the backup vector for every outcome, so a simulation never
        # allocates. Losses share the win evenly, keeping the game zero-sum for
        # any number of seats. Keyed by winning seat; -1 is a draw.
        loss = -1.0 / max(self._n_players - 1, 1)
        self._payoffs = {-1: [0.0] * self._n_players}
        for seat in range(self._n_players):
            payoff = [loss] * self._n_players
            payoff[seat] = 1.0
            self._payoffs[seat] = payoff

    # ------------------------------------------------------------------

    def select(self, state: State, actions: Actions) -> Action:
        if self.config is None or self._search is None:
            raise RuntimeError(
                "MCTSAgent needs a config; pass one to the constructor or call reset()."
            )
        position = Position(self._search, state)
        legal = position.legal_ordered(self._order)
        if not legal:
            raise ValueError("no legal actions available")
        if len(legal) == 1:
            self.last_visits = {legal[0]: 0}
            return legal[0]

        root = _Node(
            seat=position.seat,
            parent=None,
            action=None,
            untried=list(legal),
            n_players=self._n_players,
            winner_seat=-1,
            is_terminal=False,
        )

        deadline = (
            None if self.time_limit is None else time.perf_counter() + self.time_limit
        )
        simulation = 0
        while simulation < self.simulations:
            if deadline is not None and time.perf_counter() >= deadline:
                break
            self._simulate_once(position, root)
            simulation += 1

        self.last_visits = {a: c.visits for a, c in root.children.items()}
        if not root.children:
            return legal[0]
        # Most-visited is the robust choice; it is less sensitive to a single
        # lucky rollout than picking the highest mean value.
        return max(root.children.items(), key=lambda kv: kv[1].visits)[0]

    # ------------------------------------------------------------------

    def _simulate_once(self, position: Position, root: _Node) -> None:
        """One selection / expansion / playout / backup pass.

        The cursor is left exactly where it started, whatever happens.
        """
        node = root
        try:
            # Selection: descend while the node is fully expanded.
            while not node.is_terminal and not node.untried and node.children:
                node = self._best_child(node)
                action = node.action
                if action is None:  # only the root has no action
                    break
                position.push(action)

            # Expansion.
            if not node.is_terminal and node.untried:
                action = node.untried.pop(0)
                position.push(action)
                winner = position.winner_seat()
                terminal = position.terminal()
                node = self._attach(position, node, action, winner, terminal)

            # Simulation.
            if node.is_terminal:
                outcome = node.winner_seat
            else:
                result = position.playout(self.rng, self.greedy_rollouts)
                outcome = -1 if result is None else result

            self._backpropagate(node, self._payoffs[outcome])
        finally:
            position.unwind()

    def _attach(
        self,
        position: Position,
        parent: _Node,
        action: int,
        winner: int | None,
        terminal: bool,
    ) -> _Node:
        child = _Node(
            seat=position.seat,
            parent=parent,
            action=action,
            untried=[] if terminal else position.legal_ordered(self._order),
            n_players=self._n_players,
            winner_seat=-1 if winner is None else winner,
            is_terminal=terminal,
        )
        parent.children[action] = child
        return child

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

    def _backpropagate(self, node: _Node | None, payoff: list[float]) -> None:
        seats = range(self._n_players)
        while node is not None:
            node.visits += 1
            values = node.values
            for seat in seats:
                values[seat] += payoff[seat]
            node = node.parent
