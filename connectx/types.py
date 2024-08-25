from typing import TypedDict, NamedTuple

import numpy as np
import numpy.typing as npt

__all__ = ["Shape", "Config", "Grid", "Info", "State", "Action", "Actions", "Instance"]

Shape = tuple[np.uint8, np.uint8]
Config = TypedDict("Config", shape=Shape, k=np.uint8, players=list[np.uint8])

Grid = npt.NDArray[np.uint8]
Info = TypedDict("Info", active=np.uint8, time=np.uint16)
State = TypedDict("State", grid=Grid, info=Info)

Action = np.uint8
Actions = npt.NDArray[Action]

Instance = TypedDict("Instance", grid=Grid, info=Info, config=Config)
