"""Adapter tests. Skipped unless the optional ``[rl]`` extra is installed."""

import numpy as np
import pytest

from connectx import make_config, preset

pytestmark = pytest.mark.rl

gymnasium = pytest.importorskip("gymnasium", reason="needs the [rl] extra")
pettingzoo = pytest.importorskip("pettingzoo", reason="needs the [rl] extra")


class TestPettingZooEnv:
    def test_passes_the_official_api_test(self) -> None:
        from pettingzoo.test import api_test

        from connectx.adapters import make_aec_env

        api_test(make_aec_env(preset("small")), num_cycles=30, verbose_progress=False)

    def test_observation_and_mask_shapes(self) -> None:
        from connectx.adapters import make_aec_env

        config = preset("small")
        env = make_aec_env(config)
        env.reset(seed=0)
        observation = env.observe(env.agent_selection)
        rows, cols = config["shape"]
        assert observation["observation"].shape == (rows, cols, 3)
        assert observation["action_mask"].shape == (cols,)
        assert observation["action_mask"].all()

    def test_mask_closes_full_columns(self) -> None:
        from connectx.adapters import make_aec_env

        config = make_config((2, 3), 2, [1, 2])
        env = make_aec_env(config)
        env.reset(seed=0)
        env.step(0)
        env.step(0)
        assert env.observe(env.agent_selection)["action_mask"][0] == 0

    def test_perspective_is_relative_to_the_observer(self) -> None:
        from connectx.adapters import make_aec_env

        env = make_aec_env(preset("small"))
        env.reset(seed=0)
        env.step(2)
        first = env.observe("player_0")["observation"]
        second = env.observe("player_1")["observation"]
        # player_0's token sits in its own plane 0 and in player_1's plane 1.
        np.testing.assert_array_equal(first[..., 0], second[..., 1])

    def test_rewards_are_zero_sum_on_a_win(self) -> None:
        from connectx.adapters import make_aec_env

        env = make_aec_env(make_config((4, 4), 3, [1, 2]))
        env.reset(seed=0)
        for column in [0, 1, 0, 1, 0]:
            env.step(column)
        assert env.rewards["player_0"] == 1.0
        assert env.rewards["player_1"] == -1.0
        assert all(env.terminations.values())

    def test_render_returns_text(self) -> None:
        from connectx.adapters import make_aec_env

        env = make_aec_env(preset("small"), render_mode="ansi")
        env.reset(seed=0)
        assert isinstance(env.render(), str)

    def test_three_player_variant(self) -> None:
        from connectx.adapters import make_aec_env

        env = make_aec_env(make_config((5, 6), 3, [1, 2, 3]))
        env.reset(seed=0)
        assert len(env.possible_agents) == 3
        assert env.observe("player_0")["observation"].shape[-1] == 4


class TestGymEnv:
    def test_passes_the_official_env_checker(self) -> None:
        from gymnasium.utils.env_checker import check_env

        from connectx.adapters import make_gym_env

        check_env(
            make_gym_env(preset("small"), opponent="random"), skip_render_check=True
        )

    def test_episode_runs_to_termination(self) -> None:
        from connectx.adapters import make_gym_env

        env = make_gym_env(preset("small"), opponent="random")
        observation, _ = env.reset(seed=0)
        rng = np.random.default_rng(0)
        terminated = False
        steps = 0
        while not terminated and steps < 100:
            legal = np.flatnonzero(observation["action_mask"])
            observation, reward, terminated, truncated, info = env.step(
                int(rng.choice(legal))
            )
            steps += 1
        assert terminated
        assert reward in (-1.0, 0.0, 1.0)

    def test_illegal_action_ends_the_episode(self) -> None:
        from connectx.adapters import make_gym_env

        config = make_config((2, 3), 2, [1, 2])
        env = make_gym_env(config, opponent="random", illegal_move_reward=-5.0)
        env.reset(seed=0)
        env.step(0)
        env.step(0)  # column 0 now holds two tokens on a two-row board
        _, reward, terminated, _, info = env.step(0)
        assert terminated
        assert reward == -5.0
        assert info["illegal_action"] == 0

    def test_seat_one_moves_second(self) -> None:
        from connectx.adapters import make_gym_env

        env = make_gym_env(preset("small"), opponent="random", seat=1)
        observation, _ = env.reset(seed=0)
        # The opponent already moved, so the board is not empty on our first turn.
        assert observation["observation"][1].sum() == 1

    def test_rejects_a_bad_seat(self) -> None:
        from connectx.adapters import make_gym_env

        with pytest.raises(ValueError, match="seat"):
            make_gym_env(preset("small"), seat=5)

    def test_opponent_strength_is_configurable(self) -> None:
        from connectx.adapters import make_gym_env

        env = make_gym_env(preset("small"), opponent="minimax:depth=2")
        observation, _ = env.reset(seed=0)
        assert observation["action_mask"].any()

    def test_reset_is_reproducible(self) -> None:
        from connectx.adapters import make_gym_env

        def rollout(seed: int) -> list[float]:
            env = make_gym_env(preset("small"), opponent="random")
            observation, _ = env.reset(seed=seed)
            rng = np.random.default_rng(0)
            rewards = []
            terminated = False
            while not terminated:
                legal = np.flatnonzero(observation["action_mask"])
                observation, reward, terminated, _, _ = env.step(int(rng.choice(legal)))
                rewards.append(reward)
            return rewards

        assert rollout(11) == rollout(11)
