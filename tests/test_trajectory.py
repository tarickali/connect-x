import numpy as np
import pytest

from connectx import Game, ReplayMemory, Trajectory, TrajectoryStep, make_config, preset
from connectx.types import RewardSpec


def played_game(config, columns):
    game = Game(config, record=True)
    game.start()
    for column in columns:
        game.transition(column)
    return game


class TestTrajectory:
    def test_as_numpy_shapes(self) -> None:
        game = played_game(preset("connect4"), [0, 1, 0, 1, 0, 1, 0])
        packed = game.trajectory().as_numpy()
        assert packed["actions"].shape == (7,)
        assert packed["returns"].shape == (7,)
        np.testing.assert_array_equal(packed["player_index"], [0, 1, 0, 1, 0, 1, 0])
        np.testing.assert_array_equal(packed["player_token"], [1, 2, 1, 2, 1, 2, 1])

    def test_empty_trajectory(self) -> None:
        trajectory = Trajectory(config=preset("connect4"))
        packed = trajectory.as_numpy()
        assert packed["actions"].size == 0
        assert trajectory.returns().size == 0
        assert len(trajectory) == 0

    def test_returns_are_per_seat_outcomes(self) -> None:
        game = played_game(preset("connect4"), [0, 1, 0, 1, 0, 1, 0])
        # Seat 0 wins, so its moves are labelled +1 and seat 1's are -1.
        np.testing.assert_allclose(game.trajectory().returns(), [1, -1, 1, -1, 1, -1, 1])

    def test_returns_honour_the_reward_spec(self) -> None:
        game = played_game(preset("connect4"), [0, 1, 0, 1, 0, 1, 0])
        spec = RewardSpec(win=5.0, loss=-2.0, draw=0.0)
        np.testing.assert_allclose(
            game.trajectory().returns(spec), [5, -2, 5, -2, 5, -2, 5]
        )

    def test_draw_returns_are_zero(self) -> None:
        game = played_game(make_config((1, 4), 4, [1, 2]), [0, 1, 2, 3])
        np.testing.assert_allclose(game.trajectory().returns(), [0, 0, 0, 0])

    def test_replay_reconstructs_positions(self) -> None:
        config = preset("small")
        game = played_game(config, [0, 1, 2, 3, 0])
        boards = [np.array(state["grid"]) for state, _ in game.trajectory().replay()]
        assert len(boards) == 5
        assert np.all(boards[0] == 0)
        # Each replayed board should have exactly as many tokens as moves before it.
        for index, board in enumerate(boards):
            assert np.count_nonzero(board) == index

    def test_replay_matches_recorded_snapshots(self) -> None:
        config = preset("small")
        game = Game(config, record=True, record_grid_snapshots=True)
        game.start()
        for column in [0, 1, 2, 3, 1]:
            game.transition(column)
        trajectory = game.trajectory()
        replayed = [np.array(state["grid"]) for state, _ in trajectory.replay()]
        # replay() yields the board *before* each move; snapshots are after.
        for index in range(1, len(replayed)):
            np.testing.assert_array_equal(
                replayed[index], trajectory.steps[index - 1].grid_after
            )

    def test_replay_grids_are_readonly(self) -> None:
        game = played_game(preset("small"), [0, 1])
        state, _ = next(iter(game.trajectory().replay()))
        with pytest.raises(ValueError):
            state["grid"][0, 0] = 9

    def test_grids_stacks_positions(self) -> None:
        game = played_game(preset("small"), [0, 1, 2])
        grids = game.trajectory().grids()
        assert grids.shape == (3, 5, 6)


class TestReplayMemory:
    def test_push_and_cap(self) -> None:
        config = preset("connect4")
        memory = ReplayMemory(max_episodes=2)
        for _ in range(3):
            memory.push(Trajectory(config=config, steps=[]))
        assert len(memory) == 2

    def test_unbounded_by_default(self) -> None:
        config = preset("connect4")
        memory = ReplayMemory()
        for _ in range(5):
            memory.push(Trajectory(config=config, steps=[]))
        assert len(memory) == 5

    def test_steps_counts_transitions(self) -> None:
        config = preset("connect4")
        memory = ReplayMemory()
        memory.extend(
            [
                Trajectory(
                    config=config,
                    steps=[
                        TrajectoryStep(
                            action=0, player_index=0, player_token=1, time_after=1
                        )
                    ]
                    * 3,
                )
                for _ in range(4)
            ]
        )
        assert memory.steps == 12

    def test_sample_is_seeded(self) -> None:
        config = preset("connect4")
        memory = ReplayMemory()
        for index in range(10):
            memory.push(
                Trajectory(
                    config=config,
                    steps=[
                        TrajectoryStep(
                            action=index, player_index=0, player_token=1, time_after=1
                        )
                    ],
                )
            )
        first = [t.steps[0].action for t in memory.sample(4, np.random.default_rng(0))]
        second = [t.steps[0].action for t in memory.sample(4, np.random.default_rng(0))]
        assert first == second

    def test_sample_clamps_to_size(self) -> None:
        memory = ReplayMemory()
        memory.push(Trajectory(config=preset("connect4"), steps=[]))
        assert len(memory.sample(10)) == 1

    def test_sample_of_empty_memory(self) -> None:
        assert ReplayMemory().sample(3) == []

    def test_iteration_and_clear(self) -> None:
        config = preset("connect4")
        memory = ReplayMemory()
        memory.push(Trajectory(config=config, steps=[]))
        assert len(list(memory)) == 1
        assert memory[0].config == config
        memory.clear()
        assert len(memory) == 0
