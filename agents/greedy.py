"""One-ply tactical baseline: take a win, block a loss, otherwise play centre.

Useful as the second rung of the agent ladder — strong enough that beating it
means something, cheap enough to run thousands of evaluation games. It reasons
only about immediate threats, so it works for any number of players and, going
through :class:`agents.search.Position`, for any engine.
"""

from __future__ import annotations

import numpy as np

from agents.heuristics import centre_out_order
from agents.search import Position
from agents.types import BaseAgent
from connectx.engine import GameEngine
from connectx.types import Action, Actions, Config, State

__all__ = ["GreedyAgent"]


class GreedyAgent(BaseAgent):
    def __init__(self, config: Config | None = None, **kwargs) -> None:
        self._order: list[int] = []
        self._search: GameEngine | None = None
        self._probe: GameEngine | None = None
        super().__init__(config, **kwargs)

    def reset(self, config: Config) -> None:
        super().reset(config)
        # Two engines: one for our own replies, one for asking "what would an
        # opponent do from here", which needs a different seat to move.
        self._search = self.engine_factory(config, undo=True)
        self._probe = self.engine_factory(config, undo=True)
        self._order = centre_out_order(self._search.action_space_size)

    def select(self, state: State, actions: Actions) -> Action:
        if self.config is None or self._search is None or self._probe is None:
            raise RuntimeError(
                "GreedyAgent needs a config; pass one to the constructor or call reset()."
            )
        position = Position(self._search, state)
        ordered = position.legal_ordered(self._order)
        if not ordered:
            raise ValueError("no legal actions available")

        # 1. Complete a line if one is available.
        for action in ordered:
            if position.wins_immediately(action):
                return action

        # 2. Otherwise deny an opponent an immediate win. Hand the move to each
        #    opponent in turn and see whether any action wins for them; taking
        #    that action ourselves is what denies it.
        grid = np.array(position.grid, dtype=np.uint8)
        time = state["info"]["time"]
        for seat in range(position.n_players):
            if seat == position.seat:
                continue
            threat = Position(
                self._probe, {"grid": grid, "info": {"active": seat, "time": time}}
            )
            for action in ordered:
                if threat.wins_immediately(action):
                    return action

        # 3. Nothing forced: play toward the centre, breaking ties randomly.
        best = ordered[0]
        centre = (len(self._order) - 1) / 2.0
        tied = [a for a in ordered if abs(a - centre) == abs(best - centre)]
        return int(tied[self.rng.integers(len(tied))])
