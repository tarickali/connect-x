"""Turning positions into tensors a network can consume.

Two decisions here matter for generalization, which is what this project is for:

1. **Perspective-relative planes.** Plane 0 is always the player to move, plane
   1 the next player, and so on. A network therefore never has to learn "am I
   token 1 or token 2", and a single set of weights serves every seat.
2. **A fixed plane count per player-count, not per variant.** Board size shows
   up only as the spatial dimensions, so a fully-convolutional model trained on
   6x7 can be evaluated on 8x9 without touching its parameters.

Mirror symmetry is exposed separately because it is the one exact invariance
Connect-style boards have, and it doubles a dataset for free.
"""

from __future__ import annotations

import numpy as np

from connectx.types import Actions, Config, Grid, Observation, State

__all__ = [
    "observation_shape",
    "encode_grid",
    "encode_state",
    "action_mask",
    "mirror_grid",
    "mirror_action",
    "mirror_observation",
    "encode_batch",
]


def observation_shape(config: Config) -> tuple[int, int, int]:
    """``(planes, rows, cols)`` for a variant: one plane per seat, plus empties."""
    rows, cols = config["shape"]
    return (len(config["players"]) + 1, int(rows), int(cols))


def encode_grid(grid: Grid, players: list[int], perspective: int = 0) -> Observation:
    """Encode a raw grid with ``perspective``'s plane first.

    ``perspective`` is a seat index into ``players``, not a token.
    """
    n_players = len(players)
    rows, cols = grid.shape
    planes = np.zeros((n_players + 1, rows, cols), dtype=np.float32)
    for offset in range(n_players):
        seat = (perspective + offset) % n_players
        planes[offset] = grid == np.uint8(players[seat])
    planes[n_players] = grid == 0
    return planes


def encode_state(
    state: State, config: Config, *, perspective: int | None = None
) -> Observation:
    """Encode a state from ``perspective``'s point of view (default: player to move)."""
    seat = state["info"]["active"] if perspective is None else int(perspective)
    return encode_grid(state["grid"], config["players"], seat)


def action_mask(actions: Actions) -> np.ndarray:
    """Convert the ``uint8`` legality mask into ``bool``, ready to add to logits."""
    return np.asarray(actions, dtype=np.uint8).astype(bool)


def mirror_grid(grid: Grid) -> Grid:
    """Reflect a board left-to-right. The exact symmetry of Connect-style games."""
    return np.ascontiguousarray(grid[:, ::-1])


def mirror_action(action: int, cols: int) -> int:
    """The column ``action`` maps to under :func:`mirror_grid`."""
    return int(cols) - 1 - int(action)


def mirror_observation(observation: Observation) -> Observation:
    """Reflect encoded planes left-to-right, preserving plane order."""
    return np.ascontiguousarray(observation[..., ::-1])


def encode_batch(observations: np.ndarray, *, mirror: bool = False) -> np.ndarray:
    """Stack observations, optionally appending their mirror images.

    Returns ``2N`` samples when ``mirror=True``. Remember to mirror the policy
    targets with :func:`mirror_action` to match.
    """
    batch = np.asarray(observations, dtype=np.float32)
    if not mirror or batch.size == 0:
        return batch
    return np.concatenate([batch, mirror_observation(batch)], axis=0)
