"""Gymnasium single-agent adapter: you play one seat, fixed policies play the rest.

This is the shape off-the-shelf single-agent algorithms (PPO, DQN, ...) expect.
The opponent is an ordinary connect-x agent spec, so the difficulty of the
environment is a string in your experiment config — ``"random"`` to start,
``"minimax:depth=4"`` once that is solved, and a snapshot of your own policy for
self-play curricula.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "connectx.adapters.gym_env needs gymnasium. "
        "Install it with: pip install -e '.[rl]'"
    ) from exc

from connectx.config import validate_config
from connectx.encoding import encode_state
from connectx.game import Game
from connectx.renderer import grid_to_string
from connectx.types import Config, RewardSpec

__all__ = ["ConnectXGymEnv"]


class ConnectXGymEnv(gym.Env):
    """One seat of a connect-x variant, against scripted opponents."""

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(
        self,
        config: Config,
        *,
        opponent: str = "random",
        seat: int = 0,
        render_mode: str | None = None,
        rewards: RewardSpec = RewardSpec(),
        illegal_move_reward: float = -1.0,
    ) -> None:
        super().__init__()
        self.config = validate_config(config)
        self.seat = int(seat)
        self.opponent_spec = opponent
        self.render_mode = render_mode
        self.illegal_move_reward = float(illegal_move_reward)

        n_players = len(self.config["players"])
        if not 0 <= self.seat < n_players:
            raise ValueError(f"seat {seat} out of range for {n_players} players")

        self._rows, self._cols = (int(v) for v in self.config["shape"])
        self._game = Game(self.config, rewards=rewards)
        self._opponents: dict[int, Any] = {}

        planes = n_players + 1
        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(
                    low=0,
                    high=1,
                    shape=(planes, self._rows, self._cols),
                    dtype=np.float32,
                ),
                "action_mask": spaces.Box(
                    low=0, high=1, shape=(self._cols,), dtype=np.int8
                ),
            }
        )
        self.action_space = spaces.Discrete(self._cols)

    # ------------------------------------------------------------------

    def _build_opponents(self, seed: int | None) -> None:
        from agents import agent_reset, make_agent

        self._opponents = {}
        for index in range(len(self.config["players"])):
            if index == self.seat:
                continue
            agent = make_agent(
                self.opponent_spec,
                self.config,
                seed=None if seed is None else seed + 1000 + index,
            )
            agent_reset(agent, self.config)
            self._opponents[index] = agent

    def _observation(self) -> dict[str, np.ndarray]:
        state = self._game.state
        return {
            "observation": encode_state(state, self.config, perspective=self.seat).astype(
                np.float32
            ),
            "action_mask": np.asarray(self._game.actions, dtype=np.int8),
        }

    def _play_opponents(self) -> None:
        """Advance until it is our seat's turn again, or the game ends."""
        while not self._game.terminal():
            state = self._game.state
            active = state["info"]["active"]
            if active == self.seat:
                return
            action = self._opponents[active].select(state, self._game.actions)
            self._game.transition(action)

    # ------------------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict[str, np.ndarray], dict]:
        super().reset(seed=seed)
        self._build_opponents(seed)
        self._game.start()
        self._play_opponents()
        if self.render_mode == "human":
            self.render()
        return self._observation(), {"report": self._game.report()}

    def step(self, action: Any) -> tuple[dict[str, np.ndarray], float, bool, bool, dict]:
        column = int(action)
        import connectx.functional as cxf

        if not cxf.is_legal(self._game.state["grid"], column):
            # Ending the episode makes the mistake unmissable during training,
            # rather than silently wasting a step.
            return (
                self._observation(),
                self.illegal_move_reward,
                True,
                False,
                {"illegal_action": column, "report": self._game.report()},
            )

        self._game.transition(column)
        if not self._game.terminal():
            self._play_opponents()

        reward = float(self._game.rewards()[self.seat])
        terminated = self._game.terminal()
        if self.render_mode == "human":
            self.render()
        return (
            self._observation(),
            reward,
            terminated,
            False,
            {"report": self._game.report()},
        )

    def render(self) -> str | None:
        text = grid_to_string(self._game.state["grid"], k=self.config["k"])
        if self.render_mode == "human":
            print(text)
            return None
        return text

    def close(self) -> None:  # pragma: no cover - nothing to release
        pass
