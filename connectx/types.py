"""Shared types for grids, configs, states, actions, and outcomes."""

from typing import NamedTuple, TypedDict

import numpy as np
import numpy.typing as npt

__all__ = [
    "Shape",
    "Config",
    "Grid",
    "Info",
    "copy_info",
    "State",
    "Action",
    "Actions",
    "Rewards",
    "Observation",
    "Instance",
    "WinnerInfo",
    "Report",
    "StepResult",
    "RewardSpec",
]

#: ``(rows, cols)``. Plain Python ints; the engine imposes no upper bound.
Shape = tuple[int, int]


class Config(TypedDict):
    """Everything that defines a variant.

    ``players`` holds the token each seat plays; token ``0`` is reserved for
    empty cells. See :func:`connectx.config.validate_config` for the rules.
    """

    shape: Shape
    k: int
    players: list[int]


#: ``uint8`` board. ``0`` is empty, non-zero values are player tokens.
Grid = npt.NDArray[np.uint8]


class Info(TypedDict):
    """Turn bookkeeping. ``active`` indexes ``Config['players']``, not the token."""

    active: int
    time: int


def copy_info(info: Info) -> Info:
    """Copy turn counters while keeping the ``Info`` type (``dict()`` loses it)."""
    return {"active": int(info["active"]), "time": int(info["time"])}


class State(TypedDict):
    grid: Grid
    info: Info


#: A column index.
Action = int

#: Legality mask over columns, ``1`` = playable. Length equals ``shape[1]``.
Actions = npt.NDArray[np.uint8]

#: Per-seat reward, aligned with ``Config['players']``.
Rewards = npt.NDArray[np.float64]

#: Encoded board planes for neural agents, shape ``(planes, rows, cols)``.
Observation = npt.NDArray[np.float32]


class Instance(TypedDict):
    """A position plus the variant it belongs to; the unit of serialization."""

    grid: Grid
    info: Info
    config: Config


class WinnerInfo(TypedDict):
    """``token`` is the player's token, ``id`` its index in ``Config['players']``."""

    token: int
    id: int


class Report(TypedDict):
    winner: WinnerInfo | None
    steps: int
    tie: bool


class StepResult(NamedTuple):
    """What :meth:`connectx.game.Game.step` returns: the RL-shaped transition."""

    state: State
    actions: Actions
    rewards: Rewards
    terminated: bool
    report: Report


class RewardSpec(NamedTuple):
    """Terminal payoffs. Defaults are zero-sum for two players."""

    win: float = 1.0
    loss: float = -1.0
    draw: float = 0.0
