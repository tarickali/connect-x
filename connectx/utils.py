"""Construction helpers and position serialization."""

import json
from pathlib import Path

import numpy as np

from connectx.config import make_config, validate_config
from connectx.types import Config, Grid, Info, Instance, State, copy_info

__all__ = ["make_config", "make_state", "make_instance", "save", "load"]

PathLike = str | Path

_FORMAT_VERSION = 1


def make_state(grid: Grid, time: int, active: int) -> State:
    """Build a state from a grid and turn counters."""
    return {"grid": grid, "info": {"active": int(active), "time": int(time)}}


def make_instance(config: Config, state: State) -> Instance:
    """Bundle a position with the variant it belongs to."""
    return {
        "grid": state["grid"],
        "info": copy_info(state["info"]),
        "config": validate_config(config),
    }


def save(instance: Instance, filepath: PathLike) -> None:
    """Write an instance to a ``.npz`` file.

    Uses ``npz`` + JSON rather than ``pickle``: loading a position should never
    be able to execute code, and saved positions need to survive refactors of
    the Python types that produced them.
    """
    config = validate_config(instance["config"])
    meta = {
        "version": _FORMAT_VERSION,
        "config": {
            "shape": [int(config["shape"][0]), int(config["shape"][1])],
            "k": int(config["k"]),
            "players": [int(p) for p in config["players"]],
        },
        "info": {
            "active": int(instance["info"]["active"]),
            "time": int(instance["info"]["time"]),
        },
    }
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write through a handle so numpy does not append ".npz" to the given path.
    with path.open("wb") as handle:
        np.savez_compressed(
            handle,
            grid=np.ascontiguousarray(instance["grid"], dtype=np.uint8),
            meta=np.array(json.dumps(meta)),
        )


def load(filepath: PathLike) -> Instance:
    """Read an instance written by :func:`save`."""
    with np.load(Path(filepath), allow_pickle=False) as data:
        meta = json.loads(str(data["meta"].item()))
        grid = np.array(data["grid"], dtype=np.uint8)

    version = meta.get("version")
    if version != _FORMAT_VERSION:
        raise ValueError(
            f"unsupported instance format version {version!r} "
            f"(this build reads version {_FORMAT_VERSION})"
        )

    raw = meta["config"]
    config = make_config(tuple(raw["shape"]), raw["k"], raw["players"])
    info: Info = {
        "active": int(meta["info"]["active"]),
        "time": int(meta["info"]["time"]),
    }
    return {"grid": grid, "info": info, "config": config}
