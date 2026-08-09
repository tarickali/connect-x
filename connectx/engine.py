from __future__ import annotations

from typing import Protocol, runtime_checkable

from connectx.trajectory import Trajectory
from connectx.types import (
    Action,
    Actions,
    Config,
    Report,
    Rewards,
    State,
    StepResult,
)

__all__ = ["GameEngine"]


@runtime_checkable
class GameEngine(Protocol):
    """The interface agents, harnesses, and training loops program against.

    :class:`connectx.game.Game` is the drop-Connect implementation. Add games
    with different transition or termination rules — free placement, pop moves,
    obstacles, gravity changes — as separate classes satisfying this protocol.

    Constructor contract: everything in the harness builds engines as
    ``engine(config)`` or ``engine(config, record=True)``, so an implementation
    must accept a config positionally and a ``record`` keyword. Anything else it
    needs should have a default.

    ``tests/test_engine_conformance.py`` exercises this protocol; run a new
    implementation through it rather than checking by eye.
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
        """Legality mask over the action space, for the player to move."""
        ...

    @property
    def action_space_size(self) -> int:
        """Number of distinct actions, i.e. the width of a policy head.

        Drop games use one action per column; a free-placement game would use
        ``rows * cols``. Anything sizing a network or a policy target should ask
        the engine rather than assuming ``config["shape"][1]``.
        """
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

    def trajectory(self) -> Trajectory:
        """The episode recorded so far.

        Implementations built without recording should raise rather than return
        an empty episode, so a caller that wanted data finds out immediately.
        """
        ...
