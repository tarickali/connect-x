import numpy as np
import pytest

from agents import RandomAgent
from connectx.types import Actions, State
from connectx.utils import make_state


@pytest.fixture
def state_with_actions() -> tuple[State, Actions]:
    """State and valid actions (columns 0 and 1 open)."""
    grid = np.zeros((6, 7), dtype=np.uint8)
    grid[:, 2:] = 1
    actions = np.array([1, 1, 0, 0, 0, 0, 0], dtype=np.uint8)
    state = make_state(grid, time=0, active=0)
    return state, actions


class TestRandomAgent:
    def test_returns_valid_action(
        self, state_with_actions: tuple[State, Actions]
    ) -> None:
        state, actions = state_with_actions
        agent = RandomAgent()
        action = agent.select(state, actions)
        assert action in (0, 1)
        assert actions[action] == 1

    def test_deterministic_over_many_calls(
        self, state_with_actions: tuple[State, Actions]
    ) -> None:
        state, actions = state_with_actions
        agent = RandomAgent()
        chosen = [agent.select(state, actions) for _ in range(50)]
        assert all(a in (0, 1) for a in chosen)
        assert len(set(chosen)) >= 1
