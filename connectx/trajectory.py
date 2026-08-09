"""Trajectory and replay types for training and analysis.

A trajectory stores actions, not boards. Connect-style games are fully
determined by their move list, so :meth:`Trajectory.replay` can regenerate
every intermediate position on demand — an episode costs a few dozen bytes
instead of a few dozen grids. Turn on ``record_grid_snapshots`` in
:class:`connectx.game.Game` only when you need the boards materialized.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from connectx.types import Action, Config, Grid, Report, Rewards, RewardSpec, State

__all__ = ["TrajectoryStep", "Trajectory", "ReplayMemory"]


def _default_engine() -> Any:
    """Imported lazily: connectx.game imports this module."""
    from connectx.game import Game

    return Game


@dataclass
class TrajectoryStep:
    """One transition in an episode.

    ``reward`` is the immediate payoff to the player who moved, which is zero
    except on the move that ends the game. Use :meth:`Trajectory.returns` for
    the value target every seat should be trained against.
    """

    action: Action
    player_index: int
    player_token: int
    time_after: int
    reward: float = 0.0
    terminal_after: bool = False
    grid_after: Grid | None = None


@dataclass
class Trajectory:
    """A recorded episode plus the variant it was played on."""

    config: Config
    steps: list[TrajectoryStep] = field(default_factory=list)
    report: Report | None = None

    def __len__(self) -> int:
        return len(self.steps)

    @property
    def actions(self) -> np.ndarray:
        return np.array([s.action for s in self.steps], dtype=np.int64)

    def outcome_rewards(self, spec: RewardSpec = RewardSpec()) -> Rewards:
        """Terminal payoff for each seat, aligned with ``config['players']``."""
        n_players = len(self.config["players"])
        out = np.zeros(n_players, dtype=np.float64)
        if self.report is None:
            return out
        winner = self.report.get("winner")
        if winner is not None:
            out[:] = spec.loss
            out[int(winner["id"])] = spec.win
        elif self.report.get("tie"):
            out[:] = spec.draw
        return out

    def returns(self, spec: RewardSpec = RewardSpec()) -> np.ndarray:
        """Per-step value target: the final payoff of whoever moved at that step.

        This is the label an AlphaZero-style value head trains on — every
        position is scored from the perspective of the player to move.
        """
        per_seat = self.outcome_rewards(spec)
        if not self.steps:
            return np.array([], dtype=np.float64)
        return np.array([per_seat[s.player_index] for s in self.steps], dtype=np.float64)

    def replay(self, engine: Any = None) -> Iterator[tuple[State, Action]]:
        """Yield ``(state_before_move, action)`` for each step.

        Positions are rebuilt by *replaying the moves through the engine*, not
        by assuming any particular placement rule. Pass the engine the episode
        was recorded with when it is not the default drop game — reconstructing
        a free-placement episode with gravity would silently produce wrong
        boards rather than fail.
        """
        factory = engine if engine is not None else _default_engine()
        game = factory(self.config)
        state, _ = game.start()
        for step in self.steps:
            yield state, step.action
            state, _ = game.transition(step.action)

    def grids(self, engine: Any = None) -> np.ndarray:
        """Stack the position before each move as ``(steps, rows, cols)``."""
        boards = [np.array(state["grid"]) for state, _ in self.replay(engine)]
        if not boards:
            rows, cols = self.config["shape"]
            return np.zeros((0, rows, cols), dtype=np.uint8)
        return np.stack(boards)

    def as_numpy(self, spec: RewardSpec = RewardSpec()) -> dict[str, Any]:
        """Pack the episode into arrays for vectorized training code."""
        if not self.steps:
            return {
                "actions": np.array([], dtype=np.int64),
                "rewards": np.array([], dtype=np.float64),
                "returns": np.array([], dtype=np.float64),
                "player_index": np.array([], dtype=np.int64),
                "player_token": np.array([], dtype=np.int64),
            }
        return {
            "actions": self.actions,
            "rewards": np.array([s.reward for s in self.steps], dtype=np.float64),
            "returns": self.returns(spec),
            "player_index": np.array(
                [s.player_index for s in self.steps], dtype=np.int64
            ),
            "player_token": np.array(
                [s.player_token for s in self.steps], dtype=np.int64
            ),
        }


class ReplayMemory:
    """A bounded FIFO of completed trajectories (self-play or offline data)."""

    def __init__(self, max_episodes: int | None = None) -> None:
        self._episodes: list[Trajectory] = []
        self._max = max_episodes

    def push(self, episode: Trajectory) -> None:
        self._episodes.append(episode)
        if self._max is not None and len(self._episodes) > self._max:
            self._episodes.pop(0)

    def extend(self, episodes: Sequence[Trajectory]) -> None:
        for episode in episodes:
            self.push(episode)

    def sample(self, n: int, rng: np.random.Generator | None = None) -> list[Trajectory]:
        """Draw ``n`` episodes without replacement (clamped to what is stored)."""
        if not self._episodes:
            return []
        generator = np.random.default_rng() if rng is None else rng
        size = min(n, len(self._episodes))
        indices = generator.choice(len(self._episodes), size=size, replace=False)
        return [self._episodes[int(i)] for i in indices]

    @property
    def steps(self) -> int:
        """Total transitions held, which is the number a learner can batch over."""
        return sum(len(episode) for episode in self._episodes)

    def __len__(self) -> int:
        return len(self._episodes)

    def __getitem__(self, index: int) -> Trajectory:
        return self._episodes[index]

    def __iter__(self) -> Iterator[Trajectory]:
        return iter(self._episodes)

    def clear(self) -> None:
        self._episodes.clear()
