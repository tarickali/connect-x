import numpy as np
import pytest

from connectx import Game, make_config
from connectx.config import ConfigError
from connectx.types import Config, RewardSpec


def play_out(game: Game, columns) -> None:
    for column in columns:
        game.transition(column)


class TestGameStart:
    def test_returns_state_and_actions(self, default_config: Config) -> None:
        game = Game(default_config)
        state, actions = game.start()
        assert state["info"] == {"active": 0, "time": 0}
        assert actions.shape == (default_config["shape"][1],)
        assert np.all(state["grid"] == 0)

    def test_rejects_invalid_config(self) -> None:
        with pytest.raises(ConfigError):
            Game({"shape": (6, 7), "k": 99, "players": [1, 2]})  # type: ignore[arg-type]

    def test_start_from_state(self, default_config: Config) -> None:
        game = Game(default_config)
        state, _ = game.start()
        state2, _ = game.start(state=state)
        np.testing.assert_array_equal(state["grid"], state2["grid"])
        assert state2["info"] == state["info"]

    def test_resume_detects_existing_win(self) -> None:
        config = make_config((6, 7), 4, [1, 2])
        grid = np.zeros((6, 7), dtype=np.uint8)
        grid[5, 0:4] = 1
        game = Game(config)
        game.start({"grid": grid, "info": {"active": 0, "time": 4}})
        assert game.terminal()
        assert game.report()["winner"] == {"token": 1, "id": 0}

    def test_rejects_mismatched_grid_shape(self, default_config: Config) -> None:
        game = Game(default_config)
        with pytest.raises(ValueError, match="shape"):
            game.start(
                {"grid": np.zeros((3, 3), np.uint8), "info": {"active": 0, "time": 0}}
            )

    def test_rejects_out_of_range_active(self, default_config: Config) -> None:
        game = Game(default_config)
        with pytest.raises(ValueError, match="active"):
            game.start(
                {"grid": np.zeros((6, 7), np.uint8), "info": {"active": 5, "time": 0}}
            )

    def test_methods_require_start(self, default_config: Config) -> None:
        game = Game(default_config)
        for call in (
            lambda: game.state,
            lambda: game.actions,
            lambda: game.transition(0),
            lambda: game.render(),
            lambda: game.fork(),
            lambda: game.active_player,
        ):
            with pytest.raises(RuntimeError):
                call()


class TestGameTransition:
    def test_one_transition(self, default_config: Config) -> None:
        game = Game(default_config)
        game.start()
        state, _ = game.transition(0)
        assert state["info"] == {"active": 1, "time": 1}
        assert state["grid"][5, 0] == default_config["players"][0]

    def test_plays_until_terminal(self, default_config: Config) -> None:
        game = Game(default_config)
        game.start()
        moves = 0
        while not game.terminal():
            legal = np.flatnonzero(np.asarray(game.actions) == 1)
            assert legal.size > 0, "terminal() must be True when no move exists"
            state, _ = game.transition(int(legal[0]))
            moves += 1
            assert state["info"]["active"] == moves % len(default_config["players"])
        assert game.terminal()

    def test_illegal_column_raises(self, default_config: Config) -> None:
        game = Game(default_config)
        game.start()
        for _ in range(default_config["shape"][0]):
            game.transition(0)
        assert game.actions[0] == 0
        with pytest.raises(ValueError, match="Illegal action"):
            game.transition(0)

    @pytest.mark.parametrize("column", [-1, 99])
    def test_out_of_range_column_raises(self, default_config: Config, column) -> None:
        game = Game(default_config)
        game.start()
        with pytest.raises(ValueError):
            game.transition(column)

    def test_moving_after_game_over_raises(self, default_config: Config) -> None:
        game = Game(default_config)
        game.start()
        play_out(game, [0, 1, 0, 1, 0, 1, 0])
        assert game.terminal()
        with pytest.raises(RuntimeError, match="Game is over"):
            game.transition(2)


class TestGameStateIsolation:
    def test_returned_grid_is_readonly(self, default_config: Config) -> None:
        game = Game(default_config)
        state, _ = game.start()
        with pytest.raises(ValueError):
            state["grid"][0, 0] = 7

    def test_returned_actions_are_readonly(self, default_config: Config) -> None:
        game = Game(default_config)
        _, actions = game.start()
        with pytest.raises(ValueError):
            actions[0] = 0

    def test_returned_info_is_a_copy(self, default_config: Config) -> None:
        game = Game(default_config)
        state, _ = game.start()
        state["info"]["time"] = 999
        assert game.state["info"]["time"] == 0


class TestGameOutcome:
    def test_in_progress(self, default_config: Config) -> None:
        game = Game(default_config)
        game.start()
        assert game.report() == {"winner": None, "steps": 0, "tie": False}
        assert not game.rewards().any()

    def test_vertical_win_is_reported(self, default_config: Config) -> None:
        game = Game(default_config)
        game.start()
        play_out(game, [0, 1, 0, 1, 0, 1, 0])
        assert game.report() == {
            "winner": {"token": 1, "id": 0},
            "steps": 7,
            "tie": False,
        }
        np.testing.assert_allclose(game.rewards(), [1.0, -1.0])

    def test_win_that_also_fills_the_board_is_not_a_tie(self) -> None:
        # Regression: check_tie was consulted before the winner, so a winning
        # move that happened to fill the last cell was reported as a draw.
        config = make_config((1, 4), 4, [1, 2])
        game = Game(config)
        game.start(
            {
                "grid": np.array([[1, 1, 1, 0]], dtype=np.uint8),
                "info": {"active": 0, "time": 3},
            }
        )
        game.transition(3)
        report = game.report()
        assert report["tie"] is False
        assert report["winner"] == {"token": 1, "id": 0}

    def test_draw(self) -> None:
        config = make_config((1, 4), 4, [1, 2])
        game = Game(config)
        game.start()
        play_out(game, [0, 1, 2, 3])
        assert game.report()["tie"] is True
        assert game.report()["winner"] is None
        np.testing.assert_allclose(game.rewards(), [0.0, 0.0])

    def test_custom_reward_spec(self, default_config: Config) -> None:
        game = Game(default_config, rewards=RewardSpec(win=10.0, loss=0.0, draw=1.0))
        game.start()
        play_out(game, [0, 1, 0, 1, 0, 1, 0])
        np.testing.assert_allclose(game.rewards(), [10.0, 0.0])

    def test_multiplayer_rewards(self, multiplayer_config: Config) -> None:
        game = Game(multiplayer_config)
        game.start()
        while not game.terminal():
            legal = np.flatnonzero(np.asarray(game.actions) == 1)
            game.transition(int(legal[0]))
        rewards = game.rewards()
        assert rewards.shape == (3,)
        winner = game.winner()
        if winner is not None:
            assert rewards[winner["id"]] == 1.0
            assert (rewards <= 1.0).all()


class TestStepApi:
    def test_step_returns_rl_tuple(self, default_config: Config) -> None:
        game = Game(default_config)
        game.start()
        result = game.step(0)
        assert result.terminated is False
        assert result.rewards.shape == (2,)
        assert result.report["steps"] == 1
        assert result.actions.shape == (7,)

    def test_step_flags_terminal(self, default_config: Config) -> None:
        game = Game(default_config)
        game.start()
        play_out(game, [0, 1, 0, 1, 0, 1])
        result = game.step(0)
        assert result.terminated is True
        np.testing.assert_allclose(result.rewards, [1.0, -1.0])

    def test_reset_is_start(self, default_config: Config) -> None:
        game = Game(default_config)
        game.reset()
        assert game.state["info"]["time"] == 0


class TestGameUndo:
    def test_undo_restores_position(self, default_config: Config) -> None:
        game = Game(default_config, undo=True)
        game.start()
        before = np.array(game.state["grid"])
        game.transition(0)
        assert game.undo()
        np.testing.assert_array_equal(game.state["grid"], before)
        assert game.state["info"]["time"] == 0

    def test_undo_restores_terminal_status(self, default_config: Config) -> None:
        game = Game(default_config, undo=True)
        game.start()
        play_out(game, [0, 1, 0, 1, 0, 1, 0])
        assert game.terminal()
        assert game.undo()
        assert not game.terminal()
        assert game.winner() is None
        game.transition(0)
        assert game.terminal()

    def test_undo_on_fresh_game(self, default_config: Config) -> None:
        game = Game(default_config, undo=True)
        game.start()
        assert game.undo() is False

    def test_undo_disabled_by_default(self, default_config: Config) -> None:
        game = Game(default_config)
        game.start()
        game.transition(0)
        assert game.undo() is False

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
        initial = np.array(game.state["grid"])
        child = game.fork()
        child.transition(0)
        np.testing.assert_array_equal(game.state["grid"], initial)
        assert game.state["info"]["time"] == 0

    def test_fork_inherits_position(self, default_config: Config) -> None:
        game = Game(default_config)
        game.start()
        play_out(game, [3, 3, 4])
        child = game.fork()
        np.testing.assert_array_equal(child.state["grid"], game.state["grid"])
        assert child.state["info"] == game.state["info"]

    def test_fork_of_won_game_is_terminal(self, default_config: Config) -> None:
        game = Game(default_config)
        game.start()
        play_out(game, [0, 1, 0, 1, 0, 1, 0])
        assert game.fork().terminal()


class TestGameTrajectory:
    def test_records_steps(self, default_config: Config) -> None:
        game = Game(default_config, record=True)
        game.start()
        game.transition(0)
        trajectory = game.trajectory()
        assert len(trajectory.steps) == 1
        assert trajectory.steps[0].action == 0
        assert trajectory.steps[0].player_token == 1
        assert trajectory.steps[0].terminal_after == game.terminal()

    def test_records_terminal_reward(self, default_config: Config) -> None:
        game = Game(default_config, record=True)
        game.start()
        play_out(game, [0, 1, 0, 1, 0, 1, 0])
        steps = game.trajectory().steps
        assert [s.reward for s in steps[:-1]] == [0.0] * 6
        assert steps[-1].reward == 1.0

    def test_requires_record_flag(self, default_config: Config) -> None:
        game = Game(default_config)
        game.start()
        with pytest.raises(RuntimeError, match="record=True"):
            game.trajectory()

    def test_grid_snapshots_optional(self, default_config: Config) -> None:
        game = Game(default_config, record=True, record_grid_snapshots=True)
        game.start()
        game.transition(0)
        assert game.trajectory().steps[0].grid_after is not None


class TestGameInstance:
    def test_roundtrip(self, default_config: Config) -> None:
        game = Game(default_config)
        game.start()
        play_out(game, [0, 1, 2])
        restored = Game.from_instance(game.instance())
        np.testing.assert_array_equal(restored.state["grid"], game.state["grid"])
        assert restored.state["info"] == game.state["info"]


class TestGameMultiPlayer:
    def test_turn_order_and_termination(self, multiplayer_config: Config) -> None:
        game = Game(multiplayer_config)
        game.start()
        moves = 0
        while not game.terminal():
            legal = np.flatnonzero(np.asarray(game.actions) == 1)
            state, _ = game.transition(int(legal[0]))
            moves += 1
            assert state["info"]["active"] == moves % 3
        assert game.terminal()

    def test_winner_id_maps_to_token(self, multiplayer_config: Config) -> None:
        game = Game(multiplayer_config)
        game.start()
        while not game.terminal():
            legal = np.flatnonzero(np.asarray(game.actions) == 1)
            game.transition(int(legal[0]))
        winner = game.winner()
        if winner is not None:
            assert multiplayer_config["players"][winner["id"]] == winner["token"]


class TestRepr:
    def test_repr_before_and_after_start(self, default_config: Config) -> None:
        game = Game(default_config)
        assert "not started" in repr(game)
        game.start()
        assert "time=0" in repr(game)
