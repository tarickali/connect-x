"""Batched environments: many boards advanced by one call.

Self-play throughput, not per-move latency, is what bounds RL on games like
this. :class:`VecGame` keeps all boards in a single ``(n, rows, cols)`` array
and steps them inside one ``numba`` kernel, tracking column heights and the
winner incrementally so no board is ever rescanned.

Every environment in a :class:`VecGame` shares one config. Sweeping *across*
variants is :mod:`connectx.arena`'s job, not this one's.
"""

from __future__ import annotations

from typing import NamedTuple

import numba as nb
import numpy as np

from connectx.config import validate_config
from connectx.functional import winner_at
from connectx.types import Actions, Config, RewardSpec

__all__ = ["VecStepResult", "VecGame"]


class VecStepResult(NamedTuple):
    """One batched transition.

    With ``autoreset=True`` the boards of finished environments are already
    cleared in ``grids``; ``final_grids`` keeps the terminal position so a
    learner can still bootstrap from it.
    """

    grids: np.ndarray  # (n, rows, cols) uint8
    masks: np.ndarray  # (n, cols) uint8
    actives: np.ndarray  # (n,) int64 seat to move
    rewards: np.ndarray  # (n, players) float64
    terminated: np.ndarray  # (n,) bool
    final_grids: np.ndarray  # (n, rows, cols) uint8
    winners: np.ndarray  # (n,) uint8 token, 0 = draw or unfinished


@nb.njit(cache=True)
def _step_kernel(
    grids: np.ndarray,
    heights: np.ndarray,
    actives: np.ndarray,
    times: np.ndarray,
    winners: np.ndarray,
    filled: np.ndarray,
    actions: np.ndarray,
    players: np.ndarray,
    k: int,
) -> np.ndarray:
    """Apply one action per environment in place; return which ones just ended."""
    n_envs, rows, cols = grids.shape
    n_players = players.shape[0]
    cells = rows * cols
    done = np.zeros(n_envs, dtype=np.bool_)

    for env in range(n_envs):
        column = actions[env]
        if column < 0 or column >= cols or heights[env, column] >= rows:
            # Caller passed an illegal action; leave the board untouched.
            continue
        row = rows - 1 - heights[env, column]
        token = players[actives[env]]
        grids[env, row, column] = token
        heights[env, column] += 1
        filled[env] += 1

        found = winner_at(grids[env], k, row, column)
        if found != 0:
            winners[env] = found

        times[env] += 1
        # Advance from the current seat rather than the clock, matching Game.
        actives[env] = (actives[env] + 1) % n_players
        if winners[env] != 0 or filled[env] >= cells:
            done[env] = True
    return done


class VecGame:
    """``n`` independent games of the same variant, stepped together."""

    def __init__(
        self,
        config: Config,
        num_envs: int,
        *,
        autoreset: bool = True,
        rewards: RewardSpec = RewardSpec(),
    ) -> None:
        if num_envs < 1:
            raise ValueError(f"num_envs must be positive, got {num_envs}")
        self.config = validate_config(config)
        self.num_envs = int(num_envs)
        self.autoreset = bool(autoreset)
        self.reward_spec = RewardSpec(*rewards)

        self._rows, self._cols = (int(v) for v in config["shape"])
        self._k = int(config["k"])
        self._players = np.array(config["players"], dtype=np.uint8)
        self._n_players = self._players.shape[0]

        self.grids = np.zeros((self.num_envs, self._rows, self._cols), dtype=np.uint8)
        self._heights = np.zeros((self.num_envs, self._cols), dtype=np.int64)
        self.actives = np.zeros(self.num_envs, dtype=np.int64)
        self._times = np.zeros(self.num_envs, dtype=np.int64)
        self._winners = np.zeros(self.num_envs, dtype=np.uint8)
        self._filled = np.zeros(self.num_envs, dtype=np.int64)

    # ------------------------------------------------------------------

    def reset(self, indices: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Reset all environments, or just ``indices``. Returns grids and masks."""
        if indices is None:
            self.grids[:] = 0
            self._heights[:] = 0
            self.actives[:] = 0
            self._times[:] = 0
            self._winners[:] = 0
            self._filled[:] = 0
        else:
            selected = np.asarray(indices, dtype=np.int64)
            self.grids[selected] = 0
            self._heights[selected] = 0
            self.actives[selected] = 0
            self._times[selected] = 0
            self._winners[selected] = 0
            self._filled[selected] = 0
        return self.grids, self.masks()

    def masks(self) -> Actions:
        """``(n, cols)`` legality masks, derived from the tracked heights."""
        return (self._heights < self._rows).astype(np.uint8)

    def step(self, actions: np.ndarray) -> VecStepResult:
        """Advance every environment by one move."""
        columns = np.asarray(actions, dtype=np.int64).reshape(-1)
        if columns.shape[0] != self.num_envs:
            raise ValueError(f"expected {self.num_envs} actions, got {columns.shape[0]}")

        done = _step_kernel(
            self.grids,
            self._heights,
            self.actives,
            self._times,
            self._winners,
            self._filled,
            columns,
            self._players,
            self._k,
        )

        final_grids = self.grids.copy()
        winners = self._winners.copy()
        rewards = self._rewards_for(done, winners)

        if self.autoreset and done.any():
            self.reset(np.flatnonzero(done))

        return VecStepResult(
            grids=self.grids,
            masks=self.masks(),
            actives=self.actives,
            rewards=rewards,
            terminated=done,
            final_grids=final_grids,
            winners=winners,
        )

    def _rewards_for(self, done: np.ndarray, winners: np.ndarray) -> np.ndarray:
        rewards = np.zeros((self.num_envs, self._n_players), dtype=np.float64)
        if not done.any():
            return rewards
        finished = np.flatnonzero(done)
        for env in finished:
            token = int(winners[env])
            if token == 0:
                rewards[env, :] = self.reward_spec.draw
                continue
            seat = int(np.flatnonzero(self._players == np.uint8(token))[0])
            rewards[env, :] = self.reward_spec.loss
            rewards[env, seat] = self.reward_spec.win
        return rewards

    # ------------------------------------------------------------------

    def sample_actions(self, rng: np.random.Generator) -> np.ndarray:
        """Uniform random legal action per environment. Handy for smoke tests."""
        masks = self.masks()
        weights = masks.astype(np.float64)
        totals = weights.sum(axis=1, keepdims=True)
        totals[totals == 0] = 1.0
        cumulative = np.cumsum(weights / totals, axis=1)
        draws = rng.random((self.num_envs, 1))
        return np.argmax(cumulative > draws, axis=1).astype(np.int64)

    def __len__(self) -> int:
        return self.num_envs

    def __repr__(self) -> str:
        return (
            f"VecGame(num_envs={self.num_envs}, shape=({self._rows}, {self._cols}), "
            f"k={self._k}, players={list(self.config['players'])})"
        )
