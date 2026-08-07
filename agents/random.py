"""Uniform random baseline."""

from __future__ import annotations

import connectx.functional as cxf
from agents.types import BaseAgent
from connectx.types import Action, Actions, State

__all__ = ["RandomAgent"]


class RandomAgent(BaseAgent):
    """Picks uniformly among legal columns using its own seeded generator.

    Every agent owns its ``Generator`` rather than sharing ``np.random``, so
    parallel workers and repeated runs are reproducible from a single seed.
    """

    def select(self, state: State, actions: Actions) -> Action:
        columns = cxf.valid_action_columns(actions)
        if columns.size == 0:
            raise ValueError("no legal actions available")
        return int(columns[self.rng.integers(columns.size)])
