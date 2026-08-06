from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable

from connectx.types import Action, Actions, Config, State


@runtime_checkable
class GameEngine(Protocol):
    """Minimal interface for turn-based games used by agents, replay, and training loops.

    Implement this protocol in separate modules when you add games with different
    transition rules or action spaces (e.g. free placement, pop moves, obstacles).
    """

    @property
    def config(self) -> Config: ...

    def start(self, state: Optional[State] = None) -> tuple[State, Actions]:
        """Reset or resume from `state`; return observation and legal actions."""
        ...

    def reset(self, state: Optional[State] = None) -> tuple[State, Actions]:
        """Alias for `start` (RL-style naming)."""
        ...

    def step(self, action: Action) -> tuple[State, Actions]:
        """Apply one move; return next state and legal actions."""
        ...

    def terminal(self) -> bool: ...

    def legal_actions(self) -> Actions: ...

    def report(self) -> dict[str, Any]:
        """Outcome summary when `terminal` is true (or in-progress placeholder)."""
        ...
