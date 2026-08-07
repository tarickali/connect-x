"""Construction, validation, and named presets for game configs.

A :class:`connectx.types.Config` is the whole definition of a variant, so it is
also the thing a generalization experiment sweeps over. Everything that builds
one goes through :func:`validate_config` — an unvalidated config produces games
that look fine and are quietly unwinnable or unplayable.
"""

from collections.abc import Iterable, Sequence

from connectx.types import Config, Shape

__all__ = [
    "MAX_TOKEN",
    "ConfigError",
    "validate_config",
    "make_config",
    "config_id",
    "PRESETS",
    "preset",
]

#: Tokens are stored in a ``uint8`` grid, and ``0`` means empty.
MAX_TOKEN = 255


class ConfigError(ValueError):
    """Raised when a config cannot describe a playable game."""


def validate_config(config: Config) -> Config:
    """Return ``config`` unchanged, or raise :class:`ConfigError`.

    Enforces the invariants the engine relies on:

    - ``shape`` is two positive ints.
    - ``1 <= k <= max(rows, cols)``, so at least one orientation can hold a win.
    - ``players`` has at least two distinct tokens in ``1..255``; ``0`` is the
      empty-cell sentinel, and a player using it would be invisible on the board.
    """
    missing = {"shape", "k", "players"} - set(config)
    if missing:
        raise ConfigError(f"config is missing keys: {sorted(missing)}")

    shape = config["shape"]
    if not isinstance(shape, Sequence) or len(shape) != 2:
        raise ConfigError(f"shape must be a (rows, cols) pair, got {shape!r}")
    rows, cols = int(shape[0]), int(shape[1])
    if rows < 1 or cols < 1:
        raise ConfigError(f"shape must be positive in both dimensions, got {shape!r}")

    k = int(config["k"])
    if k < 1:
        raise ConfigError(f"k must be at least 1, got {k}")
    if k > max(rows, cols):
        raise ConfigError(
            f"k={k} exceeds max(rows, cols)={max(rows, cols)} for shape {(rows, cols)}: "
            "no line of that length fits, so the game can only ever end in a draw"
        )

    players = config["players"]
    if not isinstance(players, Sequence) or isinstance(players, (str, bytes)):
        raise ConfigError(f"players must be a sequence of tokens, got {players!r}")
    tokens = [int(p) for p in players]
    if len(tokens) < 2:
        raise ConfigError(f"need at least two players, got {tokens!r}")
    if len(set(tokens)) != len(tokens):
        raise ConfigError(f"player tokens must be distinct, got {tokens!r}")
    for token in tokens:
        if token < 1 or token > MAX_TOKEN:
            raise ConfigError(
                f"player token {token} out of range: tokens must be in 1..{MAX_TOKEN} "
                "(0 marks an empty cell)"
            )

    return config


def make_config(shape: Shape, k: int, players: Iterable[int]) -> Config:
    """Build and validate a config."""
    config: Config = {
        "shape": (int(shape[0]), int(shape[1])),
        "k": int(k),
        "players": [int(p) for p in players],
    }
    return validate_config(config)


def config_id(config: Config) -> str:
    """Stable short name for a variant, e.g. ``6x7k4p2``. Used in result files."""
    rows, cols = config["shape"]
    return f"{rows}x{cols}k{config['k']}p{len(config['players'])}"


#: Named variants used by the benchmarks, sweeps, and docs.
PRESETS: dict[str, Config] = {
    "tiny": make_config((4, 4), 3, [1, 2]),
    "small": make_config((5, 6), 4, [1, 2]),
    "connect4": make_config((6, 7), 4, [1, 2]),
    "connect5": make_config((9, 10), 5, [1, 2]),
    "wide": make_config((6, 12), 4, [1, 2]),
    "tall": make_config((12, 6), 4, [1, 2]),
    "three-player": make_config((7, 9), 4, [1, 2, 3]),
    "four-player": make_config((8, 10), 4, [1, 2, 3, 4]),
}


def preset(name: str) -> Config:
    """Look up a named preset, returning a fresh copy."""
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; available: {sorted(PRESETS)}")
    base = PRESETS[name]
    return make_config(base["shape"], base["k"], base["players"])
