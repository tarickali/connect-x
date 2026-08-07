"""Baseline agents and the registry the CLI and arena resolve names through.

The ladder is intentional: ``random`` < ``greedy`` < ``minimax`` < ``mcts``.
Rungs are what make a win rate interpretable — "beats random" says almost
nothing, "beats greedy 80% of the time on every variant" says a great deal.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from agents.greedy import GreedyAgent
from agents.human import HumanAgent
from agents.mcts import MCTSAgent
from agents.minimax import MinimaxAgent
from agents.random import RandomAgent
from agents.types import (
    Agent,
    BaseAgent,
    agent_name,
    agent_observe,
    agent_reset,
    agent_seed,
)
from connectx.types import Config

__all__ = [
    "Agent",
    "BaseAgent",
    "RandomAgent",
    "GreedyAgent",
    "MinimaxAgent",
    "MCTSAgent",
    "HumanAgent",
    "REGISTRY",
    "make_agent",
    "available_agents",
    "agent_name",
    "agent_reset",
    "agent_observe",
    "agent_seed",
]

REGISTRY: dict[str, Callable[..., BaseAgent]] = {
    "random": RandomAgent,
    "greedy": GreedyAgent,
    "minimax": MinimaxAgent,
    "mcts": MCTSAgent,
    "human": HumanAgent,
}


def available_agents() -> list[str]:
    return sorted(REGISTRY)


def _coerce(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def make_agent(
    spec: str, config: Config | None = None, *, seed: int | None = None
) -> BaseAgent:
    """Build an agent from a ``name:key=value,key=value`` spec.

    Examples::

        make_agent("random")
        make_agent("minimax:depth=6")
        make_agent("mcts:simulations=2000,greedy_rollouts=false")

    Keeping construction stringly-typed lets experiment configs, CLI flags, and
    result files all refer to an agent the same way.
    """
    name, _, raw_options = spec.partition(":")
    name = name.strip()
    if name not in REGISTRY:
        raise KeyError(f"unknown agent {name!r}; available: {available_agents()}")

    options: dict[str, Any] = {}
    for chunk in filter(None, (c.strip() for c in raw_options.split(","))):
        key, sep, value = chunk.partition("=")
        if not sep:
            raise ValueError(
                f"bad agent option {chunk!r} in {spec!r}; expected key=value"
            )
        options[key.strip()] = _coerce(value.strip())

    agent = REGISTRY[name](config, seed=seed, **options)
    agent._name = spec  # keep the full spec for result tables
    return agent
