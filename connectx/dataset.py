"""Persist episodes and flatten them into supervised training arrays.

Episodes are stored as move lists with an offset index, so a shard of 100k
games is a few megabytes rather than a few gigabytes of boards. Positions are
reconstructed on load via :meth:`connectx.Trajectory.replay`.

Serialization is ``npz`` + JSON, never ``pickle``: a dataset is something you
share, check into a release, or download, and none of those should be able to
execute code when opened.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

from connectx.config import make_config
from connectx.encoding import (
    encode_state,
    mirror_action,
    mirror_observation,
    observation_shape,
)
from connectx.trajectory import Trajectory, TrajectoryStep
from connectx.types import Config, RewardSpec

__all__ = ["save_trajectories", "load_trajectories", "build_supervised"]

PathLike = str | Path

_FORMAT_VERSION = 1


def save_trajectories(filepath: PathLike, trajectories: Sequence[Trajectory]) -> None:
    """Write episodes to a single ``.npz`` shard.

    All episodes in a shard must share a config; sweep results should be
    written as one shard per variant.
    """
    episodes = list(trajectories)
    if not episodes:
        raise ValueError("nothing to save: no trajectories given")

    config = episodes[0].config
    for episode in episodes[1:]:
        if (
            tuple(episode.config["shape"]) != tuple(config["shape"])
            or episode.config["k"] != config["k"]
            or list(episode.config["players"]) != list(config["players"])
        ):
            raise ValueError(
                "all trajectories in a shard must share one config; "
                "write one shard per variant"
            )

    actions: list[int] = []
    offsets = [0]
    outcomes = []
    for episode in episodes:
        actions.extend(int(step.action) for step in episode.steps)
        offsets.append(len(actions))
        report = episode.report or {"winner": None, "steps": len(episode), "tie": False}
        winner = report.get("winner")
        outcomes.append(
            [
                -1 if winner is None else int(winner["id"]),
                int(report.get("steps", len(episode))),
                1 if report.get("tie") else 0,
            ]
        )

    meta = {
        "version": _FORMAT_VERSION,
        "config": {
            "shape": [int(config["shape"][0]), int(config["shape"][1])],
            "k": int(config["k"]),
            "players": [int(p) for p in config["players"]],
        },
        "episodes": len(episodes),
    }

    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.savez_compressed(
            handle,
            actions=np.array(actions, dtype=np.int16),
            offsets=np.array(offsets, dtype=np.int64),
            outcomes=np.array(outcomes, dtype=np.int64),
            meta=np.array(json.dumps(meta)),
        )


def load_trajectories(filepath: PathLike) -> list[Trajectory]:
    """Read a shard written by :func:`save_trajectories`."""
    with np.load(Path(filepath), allow_pickle=False) as data:
        meta = json.loads(str(data["meta"].item()))
        actions = np.array(data["actions"], dtype=np.int64)
        offsets = np.array(data["offsets"], dtype=np.int64)
        outcomes = np.array(data["outcomes"], dtype=np.int64)

    if meta.get("version") != _FORMAT_VERSION:
        raise ValueError(
            f"unsupported dataset version {meta.get('version')!r} "
            f"(this build reads version {_FORMAT_VERSION})"
        )

    raw = meta["config"]
    config = make_config(tuple(raw["shape"]), raw["k"], raw["players"])
    players = config["players"]
    n_players = len(players)

    episodes: list[Trajectory] = []
    for index in range(len(offsets) - 1):
        moves = actions[offsets[index] : offsets[index + 1]]
        winner_id, steps, tie = (int(v) for v in outcomes[index])
        report = {
            "winner": (
                None
                if winner_id < 0
                else {"token": int(players[winner_id]), "id": winner_id}
            ),
            "steps": steps,
            "tie": bool(tie),
        }
        trajectory_steps = []
        for time, action in enumerate(moves):
            seat = time % n_players
            trajectory_steps.append(
                TrajectoryStep(
                    action=int(action),
                    player_index=seat,
                    player_token=int(players[seat]),
                    time_after=time + 1,
                    terminal_after=(time == len(moves) - 1),
                )
            )
        episodes.append(
            Trajectory(config=config, steps=trajectory_steps, report=report)  # type: ignore[arg-type]
        )
    return episodes


def build_supervised(
    trajectories: Iterable[Trajectory],
    *,
    spec: RewardSpec = RewardSpec(),
    mirror: bool = False,
    config: Config | None = None,
) -> dict[str, np.ndarray]:
    """Flatten episodes into per-position arrays for a policy/value learner.

    Returns ``observations`` ``(n, planes, rows, cols)``, ``policies``
    ``(n, cols)`` one-hot over the move played, ``values`` ``(n,)`` holding the
    eventual outcome for the player to move, and ``masks`` ``(n, cols)``.

    With ``mirror=True`` each position is emitted twice, the second reflected
    left-to-right with its policy target reflected to match.
    """
    observations: list[np.ndarray] = []
    policies: list[np.ndarray] = []
    values: list[float] = []
    masks: list[np.ndarray] = []
    resolved = config

    for trajectory in trajectories:
        resolved = resolved or trajectory.config
        cols = int(trajectory.config["shape"][1])
        returns = trajectory.returns(spec)
        for index, (state, action) in enumerate(trajectory.replay()):
            observation = encode_state(state, trajectory.config)
            target = np.zeros(cols, dtype=np.float32)
            target[int(action)] = 1.0
            legal = (state["grid"][0, :] == 0).astype(np.uint8)

            observations.append(observation)
            policies.append(target)
            values.append(float(returns[index]))
            masks.append(legal)

            if mirror:
                observations.append(mirror_observation(observation))
                mirrored = np.zeros(cols, dtype=np.float32)
                mirrored[mirror_action(int(action), cols)] = 1.0
                policies.append(mirrored)
                values.append(float(returns[index]))
                masks.append(legal[::-1].copy())

    if not observations:
        planes, rows, cols = observation_shape(resolved) if resolved else (1, 0, 0)
        return {
            "observations": np.zeros((0, planes, rows, cols), dtype=np.float32),
            "policies": np.zeros((0, cols), dtype=np.float32),
            "values": np.zeros((0,), dtype=np.float32),
            "masks": np.zeros((0, cols), dtype=np.uint8),
        }

    return {
        "observations": np.stack(observations).astype(np.float32),
        "policies": np.stack(policies).astype(np.float32),
        "values": np.array(values, dtype=np.float32),
        "masks": np.stack(masks).astype(np.uint8),
    }
