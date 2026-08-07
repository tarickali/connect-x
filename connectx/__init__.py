"""connect-x: parameterized Connect-style games for training and evaluating agents."""

from connectx.config import (
    PRESETS,
    ConfigError,
    config_id,
    make_config,
    preset,
    validate_config,
)
from connectx.engine import GameEngine
from connectx.game import Game
from connectx.trajectory import ReplayMemory, Trajectory, TrajectoryStep
from connectx.types import (
    Action,
    Actions,
    Config,
    Grid,
    Info,
    Instance,
    Observation,
    Report,
    Rewards,
    RewardSpec,
    Shape,
    State,
    StepResult,
    WinnerInfo,
)
from connectx.utils import load, make_instance, make_state, save

__version__ = "0.2.0"

__all__ = [
    "__version__",
    # types
    "Config",
    "State",
    "Grid",
    "Info",
    "Action",
    "Actions",
    "Rewards",
    "RewardSpec",
    "Observation",
    "Shape",
    "Instance",
    "Report",
    "StepResult",
    "WinnerInfo",
    # engine
    "Game",
    "GameEngine",
    # config
    "make_config",
    "validate_config",
    "ConfigError",
    "config_id",
    "PRESETS",
    "preset",
    # data
    "Trajectory",
    "TrajectoryStep",
    "ReplayMemory",
    "make_state",
    "make_instance",
    "save",
    "load",
]
