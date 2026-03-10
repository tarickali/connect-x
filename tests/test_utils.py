import tempfile
from pathlib import Path

import numpy as np

from connectx.types import Config, Grid
from connectx.utils import load, make_config, make_state, save


class TestMakeState:
    def test_has_grid_and_info(self) -> None:
        grid: Grid = np.zeros((6, 7), dtype=np.uint8)
        state = make_state(grid, time=0, active=1)
        assert "grid" in state
        assert "info" in state
        assert state["info"]["active"] == 1
        assert state["info"]["time"] == 0
        np.testing.assert_array_equal(state["grid"], grid)


class TestMakeConfig:
    def test_returns_dict_with_keys(self) -> None:
        c = make_config((6, 7), 4, [1, 2])
        assert c["shape"] == (6, 7)
        assert c["k"] == 4
        assert c["players"] == [1, 2]

    def test_usable_by_game(self, default_config: Config) -> None:
        c = make_config(
            default_config["shape"],
            default_config["k"],
            default_config["players"],
        )
        from connectx import Game

        game = Game(c)
        state, actions = game.start()
        assert state["grid"].shape == c["shape"]


class TestSaveLoad:
    def test_roundtrip(self, default_config: Config) -> None:
        grid: Grid = np.zeros(default_config["shape"], dtype=np.uint8)
        instance = {
            "grid": grid,
            "info": {"active": 0, "time": 0},
            "config": default_config,
        }
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            save(instance, path)
            loaded = load(path)
            np.testing.assert_array_equal(loaded["grid"], instance["grid"])
            assert loaded["info"] == instance["info"]
            assert loaded["config"] == instance["config"]
        finally:
            Path(path).unlink(missing_ok=True)
