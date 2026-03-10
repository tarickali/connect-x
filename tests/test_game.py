"""Tests for connectx.game.Game."""

import numpy as np
import pytest

from connectx import Game
from connectx.types import Config


class TestGameStart:
    def test_returns_state_and_actions(self, default_config: Config) -> None:
        game = Game(default_config)
        state, actions = game.start()
        assert "grid" in state
        assert "info" in state
        assert state["info"]["active"] == 0
        assert state["info"]["time"] == 0
        assert actions.shape == (default_config["shape"][1],)

    def test_empty_grid(self, default_config: Config) -> None:
        game = Game(default_config)
        state, _ = game.start()
        assert np.all(state["grid"] == 0)

    def test_start_from_state(self, default_config: Config) -> None:
        game = Game(default_config)
        state, _ = game.start()
        state2, _ = game.start(state=state)
        np.testing.assert_array_equal(state["grid"], state2["grid"])
        assert state2["info"] == state["info"]


class TestGameTransition:
    def test_one_transition(self, default_config: Config) -> None:
        game = Game(default_config)
        game.start()
        state, actions = game.transition(0)
        assert state["info"]["time"] == 1
        assert state["info"]["active"] == 1
        assert state["grid"][default_config["shape"][0] - 1, 0] == default_config["players"][0]

    def test_actions_updated_after_transition(self, default_config: Config) -> None:
        game = Game(default_config)
        _, actions1 = game.start()
        state, actions2 = game.transition(0)
        assert actions1.shape == actions2.shape


class TestGameTerminal:
    def test_not_terminal_at_start(self, default_config: Config) -> None:
        game = Game(default_config)
        game.start()
        assert not game.terminal()

    def test_terminal_after_vertical_win(self, default_config: Config) -> None:
        game = Game(default_config)
        game.start()
        k = default_config["k"]
        # One player must get k in a column; players alternate, so play col 0, col 1, ...
        for i in range(2 * k - 1):
            game.transition(i % 2)
        assert game.terminal()
