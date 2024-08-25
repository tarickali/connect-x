import numpy as np
from connectx.types import State, Action, Actions

__all__ = ["RandomAgent"]


class RandomAgent:
    def select(self, state: State, actions: Actions) -> Action:
        return np.random.choice(np.argwhere(actions == 1)[:, 0])
