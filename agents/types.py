"""The agent contract.

:class:`Agent` is deliberately tiny — anything with ``select`` can play. The
lifecycle hooks live on :class:`BaseAgent` and are invoked through the
``agent_*`` helpers, so a three-line agent stays valid while a learning agent
gets somewhere to receive seeds, variant changes, and outcomes.

``reset(config)`` is the important one for this project: an agent that takes
its config at construction can only ever play a single variant, which defeats
the point of measuring generalization. Agents built here re-derive whatever
they need from the config handed to them at the start of each episode.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable

import numpy as np

from connectx.types import Action, Actions, Config, State

__all__ = [
    "Agent",
    "BaseAgent",
    "agent_reset",
    "agent_observe",
    "agent_seed",
    "agent_name",
]


@runtime_checkable
class Agent(Protocol):
    """Minimum contract: given a position and a legality mask, pick a column."""

    def select(self, state: State, actions: Actions) -> Action: ...


class BaseAgent(ABC):
    """Base class supplying naming, seeding, and no-op lifecycle hooks.

    Subclasses implement :meth:`select` and override the hooks they care about.
    """

    def __init__(
        self,
        config: Config | None = None,
        *,
        seed: int | None = None,
        name: str | None = None,
        engine: Any = None,
    ) -> None:
        self._name = name or type(self).__name__
        self.config: Config | None = None
        self.rng: np.random.Generator = np.random.default_rng(seed)
        self._seed = seed
        # Search agents build a private engine to explore with. Taking it as a
        # parameter is what lets one agent play any game, rather than only the
        # drop game whose primitives it happened to import.
        self._engine_factory = engine
        if config is not None:
            self.reset(config)

    @property
    def engine_factory(self) -> Any:
        """Factory used to build private search engines. Defaults to the drop game."""
        if self._engine_factory is None:
            from agents.search import default_engine

            return default_engine()
        return self._engine_factory

    @property
    def name(self) -> str:
        return self._name

    def seed(self, seed: int | None) -> None:
        """Reseed this agent's generator. Agents never touch numpy's global RNG."""
        self._seed = seed
        self.rng = np.random.default_rng(seed)

    def reset(self, config: Config) -> None:
        """Called at the start of every episode with the variant being played."""
        self.config = config

    def observe(self, state: State, reward: float, terminal: bool) -> None:  # noqa: B027
        """Called after each of this agent's moves resolve, and at episode end.

        Deliberately concrete and empty: most agents do not learn, and forcing
        them to declare an empty override would be noise.
        """

    @abstractmethod
    def select(self, state: State, actions: Actions) -> Action: ...

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self._name!r})"


def agent_reset(agent: Agent, config: Config) -> None:
    """Call ``agent.reset(config)`` if the agent implements it."""
    hook = getattr(agent, "reset", None)
    if callable(hook):
        hook(config)


def agent_observe(agent: Agent, state: State, reward: float, terminal: bool) -> None:
    """Call ``agent.observe(...)`` if the agent implements it."""
    hook = getattr(agent, "observe", None)
    if callable(hook):
        hook(state, reward, terminal)


def agent_seed(agent: Agent, seed: int | None) -> None:
    """Call ``agent.seed(seed)`` if the agent implements it."""
    hook = getattr(agent, "seed", None)
    if callable(hook):
        hook(seed)


def agent_name(agent: Agent) -> str:
    """Best-effort display name for an arbitrary agent object."""
    name = getattr(agent, "name", None)
    if isinstance(name, str):
        return name
    return type(agent).__name__
