import pytest

from connectx.config import (
    PRESETS,
    ConfigError,
    config_id,
    make_config,
    preset,
    validate_config,
)


class TestValidateConfig:
    def test_accepts_connect4(self) -> None:
        assert validate_config({"shape": (6, 7), "k": 4, "players": [1, 2]})

    @pytest.mark.parametrize(
        "config, reason",
        [
            ({"shape": (6, 7), "k": 9, "players": [1, 2]}, "k longer than both axes"),
            ({"shape": (0, 7), "k": 4, "players": [1, 2]}, "zero rows"),
            ({"shape": (6, 0), "k": 4, "players": [1, 2]}, "zero cols"),
            ({"shape": (6, 7), "k": 0, "players": [1, 2]}, "k below one"),
            ({"shape": (6, 7), "k": 4, "players": [0, 1]}, "token 0 is the empty cell"),
            ({"shape": (6, 7), "k": 4, "players": [1, 1]}, "duplicate tokens"),
            ({"shape": (6, 7), "k": 4, "players": [1]}, "single player"),
            ({"shape": (6, 7), "k": 4, "players": [1, 256]}, "token above uint8"),
            ({"shape": (6,), "k": 4, "players": [1, 2]}, "shape not a pair"),
        ],
    )
    def test_rejects_unplayable(self, config, reason) -> None:
        with pytest.raises(ConfigError):
            validate_config(config)  # type: ignore[arg-type]

    def test_reports_missing_keys(self) -> None:
        with pytest.raises(ConfigError, match="missing keys"):
            validate_config({"shape": (6, 7)})  # type: ignore[arg-type]

    def test_k_equal_to_longest_axis_is_allowed(self) -> None:
        # A win only needs to fit along one orientation.
        assert validate_config({"shape": (2, 7), "k": 7, "players": [1, 2]})


class TestMakeConfig:
    def test_normalizes_types(self) -> None:
        config = make_config([6, 7], 4, (1, 2))
        assert config["shape"] == (6, 7)
        assert config["players"] == [1, 2]

    def test_validates(self) -> None:
        with pytest.raises(ConfigError):
            make_config((6, 7), 40, [1, 2])


class TestPresets:
    def test_all_presets_valid(self) -> None:
        for name, config in PRESETS.items():
            assert validate_config(config), name

    def test_preset_returns_a_copy(self) -> None:
        first = preset("connect4")
        first["players"].append(3)
        assert preset("connect4")["players"] == [1, 2]

    def test_unknown_preset(self) -> None:
        with pytest.raises(KeyError):
            preset("nope")


class TestConfigId:
    def test_stable_name(self) -> None:
        assert config_id(make_config((6, 7), 4, [1, 2])) == "6x7k4p2"
        assert config_id(make_config((9, 10), 5, [1, 2, 3])) == "9x10k5p3"
