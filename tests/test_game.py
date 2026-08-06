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
        assert (
            state["grid"][default_config["shape"][0] - 1, 0]
            == default_config["players"][0]
        )

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

    def test_plays_until_terminal(self, default_config: Config) -> None:
        game = Game(default_config)
        game.start()
        n_players = len(default_config["players"])
        rows, cols = default_config["shape"]
        max_moves = rows * cols + 1
        moves = 0
        while not game.terminal() and moves < max_moves:
            actions = game.actions
            valid = np.argwhere(actions == 1)[:, 0]
            assert len(valid) > 0, "terminal() should be True when no valid moves"
            state, _ = game.transition(int(valid[0]))
            moves += 1
            expected_active = moves % n_players
            assert state["info"]["active"] == expected_active
        assert game.terminal()


class TestGameIllegalMove:
    def test_illegal_column_raises(self, default_config: Config) -> None:
        game = Game(default_config)
        game.start()
        rows = int(default_config["shape"][0])
        for _ in range(rows):
            game.transition(0)
        assert game.actions[0] == 0
        with pytest.raises(ValueError):
            game.transition(0)


class TestGameReport:
    def test_not_terminal(self, default_config: Config) -> None:
        game = Game(default_config)
        game.start()
        r = game.report()
        assert r["winner"] is None
        assert r["tie"] is False
        assert r["steps"] == 0


class TestGameUndo:
    def test_undo_restores_grid(self, default_config: Config) -> None:
        game = Game(default_config, undo=True)
        game.start()
        before = np.copy(game.state["grid"])
        game.transition(0)
        assert game.undo()
        np.testing.assert_array_equal(game.state["grid"], before)

    def test_undo_pops_trajectory(self, default_config: Config) -> None:
        game = Game(default_config, undo=True, record=True)
        game.start()
        game.transition(0)
        assert len(game.trajectory().steps) == 1
        assert game.undo()
        assert len(game.trajectory().steps) == 0


class TestGameFork:
    def test_fork_parent_unchanged(self, default_config: Config) -> None:
        game = Game(default_config)
        game.start()
        initial = np.copy(game.state["grid"])
        child = game.fork()
        child.transition(0)
        np.testing.assert_array_equal(game.state["grid"], initial)


class TestGameTrajectory:
    def test_records_steps(self, default_config: Config) -> None:
        game = Game(default_config, record=True)
        game.start()
        game.transition(0)
        tr = game.trajectory()
        assert len(tr.steps) == 1
        assert tr.steps[0].action == 0
        assert tr.steps[0].terminal_after == game.terminal()


class TestGameMultiPlayer:
    def test_plays_until_terminal(self) -> None:
        config: Config = {"shape": (4, 5), "k": 3, "players": [1, 2, 3]}
        game = Game(config)
        game.start()
        n_players = len(config["players"])
        rows, cols = config["shape"]
        max_moves = rows * cols + 1
        moves = 0
        while not game.terminal() and moves < max_moves:
            actions = game.actions
            valid = np.argwhere(actions == 1)[:, 0]
            assert len(valid) > 0, "terminal() should be True when no valid moves"
            state, _ = game.transition(int(valid[0]))
            moves += 1
            expected_active = moves % n_players
            assert state["info"]["active"] == expected_active
        assert game.terminal()
