import tempfile
from pathlib import Path

import numpy as np
import pytest

from connectx import Game, preset
from connectx.types import Config, Grid
from connectx.utils import load, make_config, make_instance, make_state, save


class TestMakeState:
    def test_has_grid_and_info(self) -> None:
        grid: Grid = np.zeros((6, 7), dtype=np.uint8)
        state = make_state(grid, time=0, active=1)
        assert state["info"] == {"active": 1, "time": 0}
        np.testing.assert_array_equal(state["grid"], grid)


class TestMakeConfig:
    def test_returns_dict_with_keys(self) -> None:
        config = make_config((6, 7), 4, [1, 2])
        assert config == {"shape": (6, 7), "k": 4, "players": [1, 2]}

    def test_usable_by_game(self, default_config: Config) -> None:
        config = make_config(
            default_config["shape"], default_config["k"], default_config["players"]
        )
        state, _ = Game(config).start()
        assert state["grid"].shape == config["shape"]


class TestSaveLoad:
    def test_roundtrip(self, default_config: Config) -> None:
        grid: Grid = np.zeros(default_config["shape"], dtype=np.uint8)
        grid[5, 3] = 1
        instance = make_instance(default_config, make_state(grid, time=1, active=1))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "position.npz"
            save(instance, path)
            loaded = load(path)
        np.testing.assert_array_equal(loaded["grid"], instance["grid"])
        assert loaded["info"] == instance["info"]
        assert loaded["config"] == instance["config"]

    def test_writes_to_the_exact_path(self, default_config: Config) -> None:
        # numpy appends ".npz" when given a path; save() must not.
        instance = make_instance(
            default_config,
            make_state(np.zeros(default_config["shape"], np.uint8), 0, 0),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "no-extension"
            save(instance, path)
            assert path.exists()
            assert not Path(str(path) + ".npz").exists()

    def test_creates_parent_directories(self, default_config: Config) -> None:
        instance = make_instance(
            default_config,
            make_state(np.zeros(default_config["shape"], np.uint8), 0, 0),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "deeper" / "position.npz"
            save(instance, path)
            assert path.exists()

    def test_rejects_a_bad_version(self, default_config: Config) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.npz"
            with path.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    grid=np.zeros((6, 7), np.uint8),
                    meta=np.array('{"version": 99}'),
                )
            with pytest.raises(ValueError, match="unsupported"):
                load(path)

    def test_game_instance_roundtrip(self) -> None:
        game = Game(preset("small"))
        game.start()
        game.transition(2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "game.npz"
            save(game.instance(), path)
            restored = Game.from_instance(load(path))
        np.testing.assert_array_equal(restored.state["grid"], game.state["grid"])
        assert restored.state["info"] == game.state["info"]
