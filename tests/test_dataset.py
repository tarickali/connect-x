import tempfile
from pathlib import Path

import numpy as np
import pytest

from connectx import Game, make_config, preset
from connectx.dataset import build_supervised, load_trajectories, save_trajectories
from connectx.encoding import observation_shape
from connectx.types import RewardSpec


def random_episodes(config, count: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    game = Game(config, record=True)
    episodes = []
    for _ in range(count):
        game.start()
        while not game.terminal():
            legal = np.flatnonzero(np.asarray(game.actions) == 1)
            game.transition(int(rng.choice(legal)))
        episodes.append(game.trajectory())
    return episodes


class TestSaveLoad:
    def test_roundtrip_preserves_moves_and_outcomes(self) -> None:
        config = preset("small")
        episodes = random_episodes(config, 5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shard.npz"
            save_trajectories(path, episodes)
            restored = load_trajectories(path)
        assert len(restored) == len(episodes)
        for original, loaded in zip(episodes, restored, strict=True):
            np.testing.assert_array_equal(original.actions, loaded.actions)
            assert loaded.report == original.report
            assert loaded.config == original.config

    def test_roundtrip_preserves_returns(self) -> None:
        config = preset("small")
        episodes = random_episodes(config, 4, seed=3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shard.npz"
            save_trajectories(path, episodes)
            restored = load_trajectories(path)
        for original, loaded in zip(episodes, restored, strict=True):
            np.testing.assert_allclose(original.returns(), loaded.returns())

    def test_creates_parent_directories(self) -> None:
        config = preset("small")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "a" / "b" / "shard.npz"
            save_trajectories(path, random_episodes(config, 2))
            assert path.exists()

    def test_rejects_empty_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with pytest.raises(ValueError, match="nothing to save"):
                save_trajectories(Path(directory) / "s.npz", [])

    def test_rejects_mixed_configs(self) -> None:
        mixed = random_episodes(preset("small"), 1) + random_episodes(
            make_config((4, 5), 3, [1, 2]), 1
        )
        with tempfile.TemporaryDirectory() as directory:
            with pytest.raises(ValueError, match="share one config"):
                save_trajectories(Path(directory) / "s.npz", mixed)

    def test_rejects_a_bad_version(self) -> None:
        import json

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.npz"
            with path.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    actions=np.zeros(0, np.int16),
                    offsets=np.zeros(1, np.int64),
                    outcomes=np.zeros((0, 3), np.int64),
                    meta=np.array(json.dumps({"version": 99})),
                )
            with pytest.raises(ValueError, match="unsupported dataset version"):
                load_trajectories(path)


class TestBuildSupervised:
    def test_shapes_line_up(self) -> None:
        config = preset("small")
        episodes = random_episodes(config, 4)
        data = build_supervised(episodes)
        planes, rows, cols = observation_shape(config)
        total = sum(len(episode) for episode in episodes)
        assert data["observations"].shape == (total, planes, rows, cols)
        assert data["policies"].shape == (total, cols)
        assert data["values"].shape == (total,)
        assert data["masks"].shape == (total, cols)

    def test_policy_targets_are_one_hot_on_the_move_played(self) -> None:
        config = preset("small")
        episodes = random_episodes(config, 2)
        data = build_supervised(episodes)
        np.testing.assert_allclose(data["policies"].sum(axis=1), 1.0)
        played = np.concatenate([episode.actions for episode in episodes])
        np.testing.assert_array_equal(data["policies"].argmax(axis=1), played)

    def test_values_match_per_seat_returns(self) -> None:
        config = preset("small")
        episodes = random_episodes(config, 3)
        expected = np.concatenate([episode.returns() for episode in episodes])
        np.testing.assert_allclose(build_supervised(episodes)["values"], expected)

    def test_reward_spec_is_honoured(self) -> None:
        config = preset("small")
        episodes = random_episodes(config, 2)
        spec = RewardSpec(win=2.0, loss=-2.0, draw=0.0)
        values = build_supervised(episodes, spec=spec)["values"]
        assert set(np.unique(values)) <= {2.0, -2.0, 0.0}

    def test_mirror_doubles_and_reflects(self) -> None:
        config = preset("small")
        episodes = random_episodes(config, 2)
        plain = build_supervised(episodes)
        mirrored = build_supervised(episodes, mirror=True)
        assert mirrored["observations"].shape[0] == 2 * plain["observations"].shape[0]
        cols = config["shape"][1]
        # Every odd row is the reflection of the even row before it.
        for index in range(0, 6, 2):
            np.testing.assert_array_equal(
                mirrored["observations"][index][..., ::-1],
                mirrored["observations"][index + 1],
            )
            assert (
                mirrored["policies"][index].argmax()
                == cols - 1 - mirrored["policies"][index + 1].argmax()
            )

    def test_masks_mark_playable_columns(self) -> None:
        config = preset("small")
        data = build_supervised(random_episodes(config, 2))
        # The opening position of every game has every column open.
        assert data["masks"][0].all()

    def test_empty_input(self) -> None:
        data = build_supervised([], config=preset("small"))
        assert data["observations"].shape[0] == 0
        assert data["values"].shape == (0,)
