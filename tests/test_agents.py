import numpy as np
import pytest

from agents import RandomAgent, MinimaxAgent
from connectx.types import Actions, State, Config
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


class TestMinimaxAgent:
    def _default_config(self) -> Config:
        return {"shape": (6, 7), "k": 4, "players": [1, 2]}

    def test_takes_immediate_win(self) -> None:
        config = self._default_config()
        # Set up board for immediate win
        grid = np.zeros(config["shape"], dtype=np.uint8)
        grid[config["shape"][0] - 1, 0:3] = 1
        # Generate all available actions
        actions = np.ones(config["shape"][1], dtype=np.uint8)
        # Create a state for the agent
        state = make_state(grid, time=3, active=0)

        agent = MinimaxAgent(config=config, depth=2)
        action = int(agent.select(state, actions))
        assert action == 3  # complete the connect-4

    def test_blocks_opponent_win(self) -> None:
        config = self._default_config()
        # Set up board for immediate opponent win
        grid = np.zeros(config["shape"], dtype=np.uint8)
        grid[config["shape"][0] - 1, 0:3] = 2
        # Generate all available actions
        actions = np.ones(config["shape"][1], dtype=np.uint8)
        # Create a state for the agent
        state = make_state(grid, time=3, active=0)

        agent = MinimaxAgent(config=config, depth=3)
        action = int(agent.select(state, actions))
        assert action == 3  # block opponent's connect-4

    def test_depth_uses_evaluate(self) -> None:
        config = self._default_config()
        grid = np.zeros(config["shape"], dtype=np.uint8)
        actions = np.ones(config["shape"][1], dtype=np.uint8)
        state = make_state(grid, time=0, active=0)
        agent = MinimaxAgent(config=config, depth=1)
        action = int(agent.select(state, actions))
        assert 0 <= action < config["shape"][1]
