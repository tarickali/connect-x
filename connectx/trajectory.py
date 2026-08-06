"""Trajectory and replay types for training and analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np

from connectx.types import Config, Grid

__all__ = ["TrajectoryStep", "Trajectory", "ReplayMemory"]


@dataclass
class TrajectoryStep:
    """One transition in an episode."""

    action: int
    player_index: int
    player_token: int
    time_after: int
    reward: float = 0.0
    terminal_after: bool = False
    grid_after: Optional[Grid] = None


@dataclass
class Trajectory:
    config: Config
    steps: List[TrajectoryStep] = field(default_factory=list)

    def as_numpy(self) -> dict[str, Any]:
        """Pack actions and rewards into arrays for vectorized training code."""
        if not self.steps:
            return {
                "actions": np.array([], dtype=np.int64),
                "rewards": np.array([], dtype=np.float64),
                "player_index": np.array([], dtype=np.int64),
            }
        return {
            "actions": np.array([s.action for s in self.steps], dtype=np.int64),
            "rewards": np.array([s.reward for s in self.steps], dtype=np.float64),
            "player_index": np.array([s.player_index for s in self.steps], dtype=np.int64),
        }


class ReplayMemory:
    """Stores completed trajectories (e.g. self-play or offline datasets)."""

    def __init__(self, max_episodes: Optional[int] = None) -> None:
        self._episodes: List[Trajectory] = []
        self._max = max_episodes

    def push(self, episode: Trajectory) -> None:
        self._episodes.append(episode)
        if self._max is not None and len(self._episodes) > self._max:
            self._episodes.pop(0)

    def __len__(self) -> int:
        return len(self._episodes)

    def __getitem__(self, index: int) -> Trajectory:
        return self._episodes[index]

    def __iter__(self):
        return iter(self._episodes)

    def clear(self) -> None:
        self._episodes.clear()
