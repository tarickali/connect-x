from __future__ import annotations

from typing import Protocol, runtime_checkable

from connectx.types import Action, Actions, Config, Report, Rewards, State, StepResult

__all__ = ["GameEngine"]


@runtime_checkable
class GameEngine(Protocol):
    """The interface agents, harnesses, and training loops program against.

    :class:`connectx.game.Game` is the drop-Connect implementation. Add games
    with different transition or termination rules — free placement, pop moves,
    obstacles, gravity changes — as separate classes satisfying this protocol,
    and everything downstream (arena, adapters, recorders) keeps working.
    """

    @property
    def config(self) -> Config:
        """The variant being played."""
        ...

    @property
    def state(self) -> State:
        """Current position. Implementations should return a non-writeable grid."""
        ...

    @property
    def actions(self) -> Actions:
        """Legality mask over columns for the player to move."""
        ...

    def start(self, state: State | None = None) -> tuple[State, Actions]:
        """Reset, or resume from ``state``; return the position and legal actions."""
        ...

    def reset(self, state: State | None = None) -> tuple[State, Actions]:
        """Alias for :meth:`start` (RL-style naming)."""
        ...

    def transition(self, action: Action) -> tuple[State, Actions]:
        """Apply one move; return the next position and legal actions."""
        ...

    def step(self, action: Action) -> StepResult:
        """Apply one move; return position, actions, rewards, done, and report."""
        ...

    def terminal(self) -> bool:
        """Whether the episode has ended."""
        ...

    def legal_actions(self) -> Actions:
        """Same as :attr:`actions`, as a method for callers that prefer one."""
        ...

    def rewards(self) -> Rewards:
        """Per-seat reward, aligned with ``config['players']``; zeros mid-game."""
        ...

    def report(self) -> Report:
        """Outcome summary; empty winner and tie while the game is in progress."""
        ...
