"""PettingZoo AEC adapter.

AEC (agent-environment-cycle) is the right abstraction for a turn-based game:
one agent acts at a time and the environment says who is next, which is exactly
what :class:`connectx.Game` already does. Observations follow the convention of
PettingZoo's own classic environments — channels-last planes plus an
``action_mask`` — so wrappers and example scripts written against those work
here too.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import gymnasium.spaces as spaces
    import pettingzoo.utils as pz_utils
    from pettingzoo import AECEnv
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "connectx.adapters.pettingzoo_env needs pettingzoo and gymnasium. "
        "Install them with: pip install -e '.[rl]'"
    ) from exc

# PettingZoo renamed `agent_selector` to `AgentSelector` in 1.25; support both
# so the adapter is not pinned to one release of an actively-moving library.
AgentSelector = getattr(pz_utils, "AgentSelector", None) or pz_utils.agent_selector

from connectx.config import validate_config  # noqa: E402
from connectx.encoding import encode_state  # noqa: E402
from connectx.game import Game  # noqa: E402
from connectx.renderer import grid_to_string  # noqa: E402
from connectx.types import Config, RewardSpec  # noqa: E402

__all__ = ["ConnectXAECEnv", "env", "raw_env"]


class ConnectXAECEnv(AECEnv):
    """Turn-based multi-agent view of a connect-x variant."""

    metadata = {
        "render_modes": ["human", "ansi"],
        "name": "connectx_v0",
        "is_parallelizable": False,
    }

    def __init__(
        self,
        config: Config,
        *,
        render_mode: str | None = None,
        rewards: RewardSpec = RewardSpec(),
        engine: Any = None,
    ) -> None:
        super().__init__()
        self.config = validate_config(config)
        self.render_mode = render_mode
        factory = engine if engine is not None else Game
        self._game = factory(self.config, rewards=rewards)
        self._rows, self._cols = (int(v) for v in self.config["shape"])
        self._n_players = len(self.config["players"])

        self.possible_agents = [f"player_{i}" for i in range(self._n_players)]
        self.agents = list(self.possible_agents)

        planes = self._n_players + 1
        self._observation_spaces: dict[str, spaces.Space] = {
            agent: spaces.Dict(
                {
                    "observation": spaces.Box(
                        low=0,
                        high=1,
                        shape=(self._rows, self._cols, planes),
                        dtype=np.float32,
                    ),
                    "action_mask": spaces.Box(
                        low=0, high=1, shape=(self._cols,), dtype=np.int8
                    ),
                }
            )
            for agent in self.possible_agents
        }
        self._action_spaces: dict[str, spaces.Space] = {
            agent: spaces.Discrete(self._cols) for agent in self.possible_agents
        }

    # ------------------------------------------------------------------

    def observation_space(self, agent: str) -> spaces.Space:
        return self._observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Space:
        return self._action_spaces[agent]

    def _seat(self, agent: str) -> int:
        return self.possible_agents.index(agent)

    def observe(self, agent: str) -> dict[str, np.ndarray]:
        """Planes from ``agent``'s perspective, plus the legality mask."""
        state = self._game.state
        planes = encode_state(state, self.config, perspective=self._seat(agent))
        mask = np.asarray(self._game.actions, dtype=np.int8)
        if self.terminations[agent] or self.truncations[agent]:
            mask = np.zeros(self._cols, dtype=np.int8)
        return {
            # Channels-last to match PettingZoo's classic environments.
            "observation": np.transpose(planes, (1, 2, 0)).astype(np.float32),
            "action_mask": mask,
        }

    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None, options: dict | None = None) -> None:
        self.agents = list(self.possible_agents)
        self._game.start()
        self._agent_selector = AgentSelector(self.agents)
        self.agent_selection = self._agent_selector.reset()
        self.rewards = dict.fromkeys(self.agents, 0.0)
        self._cumulative_rewards = dict.fromkeys(self.agents, 0.0)
        self.terminations = dict.fromkeys(self.agents, False)
        self.truncations = dict.fromkeys(self.agents, False)
        self.infos: dict[str, dict] = {agent: {} for agent in self.agents}
        if self.render_mode == "human":
            self.render()

    def step(self, action: Any) -> None:
        agent = self.agent_selection
        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return

        self._cumulative_rewards[agent] = 0.0
        self._game.transition(int(action))

        payoffs = self._game.rewards()
        self.rewards = {name: float(payoffs[self._seat(name)]) for name in self.agents}
        if self._game.terminal():
            self.terminations = dict.fromkeys(self.agents, True)

        self.agent_selection = self._agent_selector.next()
        self._accumulate_rewards()

        if self.render_mode == "human":
            self.render()

    def render(self) -> str | None:
        text = grid_to_string(self._game.state["grid"], k=self.config["k"])
        if self.render_mode == "human":
            print(text)
            return None
        return text

    def close(self) -> None:  # pragma: no cover - nothing to release
        pass


def raw_env(config: Config, **kwargs) -> ConnectXAECEnv:
    """PettingZoo naming convention for the unwrapped environment."""
    return ConnectXAECEnv(config, **kwargs)


def env(config: Config, **kwargs) -> AECEnv:
    """Wrapped environment with PettingZoo's standard error-checking wrappers."""
    from pettingzoo.utils import wrappers

    environment = raw_env(config, **kwargs)
    environment = wrappers.AssertOutOfBoundsWrapper(environment)
    environment = wrappers.OrderEnforcingWrapper(environment)
    return environment
