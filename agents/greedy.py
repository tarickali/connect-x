"""One-ply tactical baseline: take a win, block a loss, otherwise play centre.

Useful as the second rung of the agent ladder — strong enough that beating it
means something, cheap enough to run thousands of evaluation games. Unlike
:class:`~agents.minimax.MinimaxAgent` it handles any number of players, since
it only ever reasons about immediate threats.
"""

from __future__ import annotations

import numpy as np

import connectx.functional as cxf
from agents.heuristics import column_order
from agents.types import BaseAgent
from connectx.types import Action, Actions, Config, State

__all__ = ["GreedyAgent"]


class GreedyAgent(BaseAgent):
    def __init__(self, config: Config | None = None, **kwargs) -> None:
        self._order: np.ndarray = np.zeros(0, dtype=np.int64)
        super().__init__(config, **kwargs)

    def reset(self, config: Config) -> None:
        super().reset(config)
        self._order = column_order(int(config["shape"][1]))

    def select(self, state: State, actions: Actions) -> Action:
        if self.config is None:
            raise RuntimeError(
                "GreedyAgent needs a config; pass one to the constructor or call reset()."
            )
        grid = state["grid"]
        k = int(self.config["k"])
        players = self.config["players"]
        me = np.uint8(players[state["info"]["active"]])

        legal = {int(c) for c in cxf.valid_action_columns(actions)}
        if not legal:
            raise ValueError("no legal actions available")
        ordered = [int(c) for c in self._order if int(c) in legal]

        # 1. Complete a line if one is available.
        for column in ordered:
            row = cxf.drop_row(grid, column)
            child = cxf.place_token(grid, me, column)
            if cxf.winner_at(child, k, row, column) == me:
                return column

        # 2. Otherwise deny any opponent an immediate win.
        for column in ordered:
            row = cxf.drop_row(grid, column)
            for token in players:
                if int(token) == int(me):
                    continue
                threat = cxf.place_token(grid, np.uint8(token), column)
                if cxf.winner_at(threat, k, row, column) == np.uint8(token):
                    return column

        # 3. Nothing forced: play toward the centre, breaking ties randomly.
        best = ordered[0]
        center = (grid.shape[1] - 1) / 2.0
        tied = [c for c in ordered if abs(c - center) == abs(best - center)]
        return int(tied[self.rng.integers(len(tied))])
