import numpy as np
import pytest

import connectx.functional as cxf
from connectx import Game, make_config, preset
from connectx.types import RewardSpec
from connectx.vector import VecGame


class TestVecGameBasics:
    def test_rejects_zero_envs(self) -> None:
        with pytest.raises(ValueError):
            VecGame(preset("small"), 0)

    def test_reset_shapes(self) -> None:
        vec = VecGame(preset("connect4"), 8)
        grids, masks = vec.reset()
        assert grids.shape == (8, 6, 7)
        assert masks.shape == (8, 7)
        assert (masks == 1).all()
        assert len(vec) == 8

    def test_action_count_is_checked(self) -> None:
        vec = VecGame(preset("small"), 4)
        vec.reset()
        with pytest.raises(ValueError, match="expected 4 actions"):
            vec.step(np.zeros(3, dtype=np.int64))

    def test_masks_track_column_height(self) -> None:
        config = make_config((3, 4), 3, [1, 2])
        vec = VecGame(config, 1, autoreset=False)
        vec.reset()
        for _ in range(3):
            vec.step(np.array([0]))
        assert vec.masks()[0, 0] == 0
        assert vec.masks()[0, 1] == 1

    def test_illegal_actions_are_ignored(self) -> None:
        config = make_config((2, 3), 2, [1, 2])
        vec = VecGame(config, 1, autoreset=False)
        vec.reset()
        before = vec.grids.copy()
        vec.step(np.array([99]))
        np.testing.assert_array_equal(vec.grids, before)


class TestVecGameSemantics:
    def test_matches_the_scalar_engine(self) -> None:
        """A batched rollout must agree move-for-move with Game."""
        config = preset("small")
        rng = np.random.default_rng(0)
        for _ in range(20):
            vec = VecGame(config, 1, autoreset=False)
            vec.reset()
            game = Game(config)
            game.start()
            while not game.terminal():
                legal = cxf.valid_action_columns(game.actions)
                column = int(rng.choice(legal))
                game.transition(column)
                result = vec.step(np.array([column]))
                np.testing.assert_array_equal(vec.grids[0], game.state["grid"])
                assert bool(result.terminated[0]) == game.terminal()
            winner = game.winner()
            expected = 0 if winner is None else winner["token"]
            assert int(result.winners[0]) == expected

    def test_rewards_are_zero_sum_on_a_win(self) -> None:
        config = make_config((4, 4), 3, [1, 2])
        vec = VecGame(config, 1, autoreset=False)
        vec.reset()
        for column in [0, 1, 0, 1, 0]:
            result = vec.step(np.array([column]))
        assert bool(result.terminated[0])
        np.testing.assert_allclose(result.rewards[0], [1.0, -1.0])

    def test_draw_rewards(self) -> None:
        config = make_config((1, 4), 4, [1, 2])
        vec = VecGame(config, 1, autoreset=False)
        vec.reset()
        for column in range(4):
            result = vec.step(np.array([column]))
        assert bool(result.terminated[0])
        np.testing.assert_allclose(result.rewards[0], [0.0, 0.0])

    def test_custom_reward_spec(self) -> None:
        config = make_config((4, 4), 3, [1, 2])
        vec = VecGame(config, 1, autoreset=False, rewards=RewardSpec(3.0, -3.0, 0.5))
        vec.reset()
        for column in [0, 1, 0, 1, 0]:
            result = vec.step(np.array([column]))
        np.testing.assert_allclose(result.rewards[0], [3.0, -3.0])


class TestVecGameAutoreset:
    def test_finished_envs_are_cleared(self) -> None:
        config = make_config((4, 4), 3, [1, 2])
        vec = VecGame(config, 1, autoreset=True)
        vec.reset()
        for column in [0, 1, 0, 1, 0]:
            result = vec.step(np.array([column]))
        assert bool(result.terminated[0])
        # The live board is already reset; the terminal one is preserved.
        assert vec.grids[0].sum() == 0
        assert result.final_grids[0].sum() > 0

    def test_no_autoreset_keeps_the_board(self) -> None:
        config = make_config((4, 4), 3, [1, 2])
        vec = VecGame(config, 1, autoreset=False)
        vec.reset()
        for column in [0, 1, 0, 1, 0]:
            vec.step(np.array([column]))
        assert vec.grids[0].sum() > 0

    def test_partial_reset(self) -> None:
        vec = VecGame(preset("small"), 4, autoreset=False)
        vec.reset()
        vec.step(np.zeros(4, dtype=np.int64))
        vec.reset(np.array([0, 2]))
        assert vec.grids[0].sum() == 0
        assert vec.grids[1].sum() == 1
        assert vec.grids[2].sum() == 0

    def test_long_run_stays_consistent(self) -> None:
        config = preset("small")
        rng = np.random.default_rng(1)
        vec = VecGame(config, 32)
        vec.reset()
        finished = 0
        for _ in range(400):
            result = vec.step(vec.sample_actions(rng))
            finished += int(result.terminated.sum())
            # Every live board must have a legal move available.
            assert (vec.masks().sum(axis=1) > 0).all()
        assert finished > 0

    def test_sample_actions_are_legal(self) -> None:
        config = make_config((2, 3), 2, [1, 2])
        rng = np.random.default_rng(0)
        vec = VecGame(config, 16, autoreset=False)
        vec.reset()
        for _ in range(3):
            actions = vec.sample_actions(rng)
            masks = vec.masks()
            for env, column in enumerate(actions):
                if masks[env].any():
                    assert masks[env, column] == 1
            vec.step(actions)


class TestVecGameMultiplayer:
    def test_three_seats_rotate(self) -> None:
        config = make_config((5, 6), 3, [1, 2, 3])
        vec = VecGame(config, 2, autoreset=False)
        vec.reset()
        assert (vec.actives == 0).all()
        vec.step(np.zeros(2, dtype=np.int64))
        assert (vec.actives == 1).all()
        vec.step(np.ones(2, dtype=np.int64))
        assert (vec.actives == 2).all()
        result = vec.step(np.full(2, 2, dtype=np.int64))
        assert result.rewards.shape == (2, 3)
