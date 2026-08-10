"""An engine-agnostic cursor for tree search.

Search agents used to reach past the engine and call the drop primitives
directly, which meant they could only ever play drop games — a new engine would
pass the conformance suite and still have no baseline able to play it.

:class:`Position` fixes that. It wraps any :class:`~connectx.engine.GameEngine`
and exposes only what a search actually needs: enumerate moves, apply one,
recurse, take it back. Nothing here knows how a move is applied, so the same
minimax and MCTS run on free placement, captures, or pop moves.

Two fast paths are used when the engine offers them, and skipped when it does
not:

- ``undo()`` for take-back. The fallback is snapshot-and-restore through
  ``start(state)``, which is correct but rescans the board.
- ``rollout()`` for random playouts. The fallback steps the engine move by move.

An engine that implements neither still works; it is only slower.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from connectx.types import Config, Grid, State, copy_info

__all__ = ["Position", "default_engine"]


def default_engine() -> Any:
    """The drop game, imported lazily to keep `agents` importable on its own."""
    from connectx.game import Game

    return Game


class Position:
    """A movable cursor over an engine's game tree.

    Invariant: every :meth:`push` must be matched by a :meth:`pop`, and the
    engine is left exactly as it was found.
    """

    __slots__ = (
        "_engine",
        "_can_undo",
        "_can_rollout",
        "_stack",
        "_state",
        "_actions",
        "_depth",
        "players",
        "n_players",
        "k",
    )

    def __init__(self, engine: Any, state: State | None = None) -> None:
        self._engine = engine
        config: Config = engine.config
        self.players = [int(p) for p in config["players"]]
        self.n_players = len(self.players)
        self.k = int(config["k"])

        self._can_undo = callable(getattr(engine, "undo", None))
        self._can_rollout = callable(getattr(engine, "rollout", None))
        # Each entry is the (state, actions) pair we came from, plus a snapshot
        # for engines without undo. Restoring the cached pair on pop avoids two
        # defensive property reads per node, which is measurable in search.
        self._stack: list[tuple[State, Any, State | None]] = []
        self._depth = 0

        self._state, self._actions = engine.start(state)

    # ------------------------------------------------------------------
    # Reading the current position
    # ------------------------------------------------------------------

    @property
    def grid(self) -> Grid:
        return self._state["grid"]

    @property
    def seat(self) -> int:
        """Index of the player to move."""
        return self._state["info"]["active"]

    @property
    def token(self) -> int:
        return self.players[self._state["info"]["active"]]

    @property
    def depth(self) -> int:
        """How many un-popped pushes deep we are."""
        return self._depth

    def opponent_of(self, seat: int) -> int:
        """The other seat, for two-player search."""
        return 1 - seat if self.n_players == 2 else (seat + 1) % self.n_players

    def legal(self) -> list[int]:
        """Legal actions, in ascending order."""
        mask = self._actions
        return [index for index in range(len(mask)) if mask[index]]

    def legal_ordered(self, order: list[int]) -> list[int]:
        """Legal actions, in the caller's preferred order (for move ordering)."""
        mask = self._actions
        return [index for index in order if index < len(mask) and mask[index]]

    def terminal(self) -> bool:
        return self._engine.terminal()

    def winner_seat(self) -> int | None:
        """Seat that has won, or None for a draw or an unfinished game."""
        winner = self._engine.report()["winner"]
        return None if winner is None else int(winner["id"])

    # ------------------------------------------------------------------
    # Moving through the tree
    # ------------------------------------------------------------------

    def push(self, action: int) -> None:
        """Apply ``action``. Undo it with :meth:`pop`."""
        snapshot: State | None = None
        if not self._can_undo:
            # The grid must be copied: the engine is free to hand out views of
            # a buffer it will replace.
            snapshot = {
                "grid": np.array(self._state["grid"], dtype=np.uint8),
                "info": copy_info(self._state["info"]),
            }
        self._stack.append((self._state, self._actions, snapshot))
        self._state, self._actions = self._engine.transition(action)
        self._depth += 1

    def pop(self) -> None:
        """Take back the last :meth:`push`."""
        if self._depth == 0:
            raise RuntimeError("pop() without a matching push()")
        state, actions, snapshot = self._stack.pop()
        if self._can_undo:
            self._engine.undo()
            # The engine is back where it was, so the pair we cached on the way
            # in still describes it exactly.
            self._state, self._actions = state, actions
        else:
            self._state, self._actions = self._engine.start(snapshot)
        self._depth -= 1

    def unwind(self) -> None:
        """Pop everything, returning to where the cursor started."""
        while self._depth:
            self.pop()

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def wins_immediately(self, action: int) -> bool:
        """Whether ``action`` ends the game in favour of the player to move."""
        seat = self.seat
        self.push(action)
        try:
            return self.winner_seat() == seat
        finally:
            self.pop()

    def playout(self, rng: np.random.Generator, greedy: bool = True) -> int | None:
        """Play to the end at random; return the winning seat, or None for a draw.

        The position is restored before returning. Uses the engine's compiled
        rollout when it has one, and walks the tree otherwise.
        """
        if self._can_rollout:
            token = int(
                self._engine.rollout(seed=int(rng.integers(1, 2**31 - 1)), greedy=greedy)
            )
            if token == 0:
                return None
            return self.players.index(token)

        started = self._depth
        while not self.terminal():
            options = self.legal()
            if not options:
                break
            choice = -1
            if greedy:
                for action in options:
                    if self.wins_immediately(action):
                        choice = action
                        break
            if choice < 0:
                choice = options[int(rng.integers(len(options)))]
            self.push(choice)
        winner = self.winner_seat()
        while self._depth > started:
            self.pop()
        return winner
