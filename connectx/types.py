from typing import TypedDict

import numpy as np
import numpy.typing as npt

__all__ = ["Shape", "Config", "Grid", "Info", "State", "Action", "Actions", "Instance"]

Shape = tuple[np.uint8, np.uint8]


class Config(TypedDict):
    shape: Shape
    k: int
    players: list


Grid = npt.NDArray[np.uint8]


class Info(TypedDict):
    active: int
    time: int


class State(TypedDict):
    grid: Grid
    info: Info


Action = np.uint8
Actions = npt.NDArray[Action]


class Instance(TypedDict):
    grid: Grid
    info: Info
    config: Config
