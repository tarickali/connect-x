"""A free-placement engine, used to prove the rest of the project is general.

This is a **test fixture, not a shipped game**. It exists so the conformance
suite and the agent ladder are exercised against dynamics that differ from the
drop game in the ways that matter:

- **No gravity.** A token goes wherever you put it.
- **A different action space.** ``rows * cols`` actions rather than one per
  column, so anything that assumed "action index == column" breaks here.
- **No compiled rollout.** :class:`agents.search.Position` must fall back to
  stepping the engine for Monte Carlo playouts.

That makes it the general m,n,k game — tic-tac-toe at 3x3 k=3, Gomoku at 15x15
k=5. A real implementation would want a compiled core and an incremental legal
move list; this one is written for clarity so a reviewer can check it by eye.

The win test is :func:`connectx.functional.winner_at`, unchanged: checking the
four lines through the cell just filled is a statement about lines, not about
how the token got there.
"""

from __future__ import annotations

import numpy as np

import connectx.functional as cxf
from connectx.config import validate_config
from connectx.trajectory import Trajectory, TrajectoryStep
from connectx.types import (
    Action,
    Actions,
    Config,
    Grid,
    Info,
    Report,
    Rewards,
    RewardSpec,
    State,
    StepResult,
    WinnerInfo,
    copy_info,
)

__all__ = ["FreePlacementGame"]


class FreePlacementGame:
    """m,n,k game with free placement. Implements ``connectx.engine.GameEngine``."""

    def __init__(
        self,
        config: Config,
        *,
        undo: bool = False,
        record: bool = False,
        rewards: RewardSpec = RewardSpec(),
    ) -> None:
        self._config = validate_config(config)
        self._rewards = RewardSpec(*rewards)
        self._rows, self._cols = (int(v) for v in config["shape"])
        self._k = int(config["k"])
        self._grid: Grid | None = None
        self._info: Info | None = None
        self._winner_token = 0
        self._filled = 0
        self._cells = self._rows * self._cols
        # `undo` is accepted for signature compatibility and ignored: this
        # engine always keeps an undo stack, since it snapshots the board on
        # every move regardless.
        del undo
        self._undo_stack: list[tuple[Grid, Info, int, int]] = []
        self._steps: list[TrajectoryStep] | None = [] if record else None

    # ------------------------------------------------------------------

    @property
    def config(self) -> Config:
        return self._config

    @property
    def action_space_size(self) -> int:
        return self._cells

    def _coords(self, action: int) -> tuple[int, int]:
        return divmod(int(action), self._cols)

    def start(self, state: State | None = None) -> tuple[State, Actions]:
        if state is None:
            self._grid = np.zeros((self._rows, self._cols), dtype=np.uint8)
            self._info = {"active": 0, "time": 0}
            self._winner_token = 0
            self._filled = 0
            self._undo_stack.clear()
            if self._steps is not None:
                self._steps.clear()
        else:
            grid = np.array(state["grid"], dtype=np.uint8)
            if grid.shape != (self._rows, self._cols):
                raise ValueError(f"state grid has shape {grid.shape}")
            active = int(state["info"]["active"])
            if not 0 <= active < len(self._config["players"]):
                raise ValueError(f"state active index {active} out of range")
            self._grid = grid
            self._info = {"active": active, "time": int(state["info"]["time"])}
            self._winner_token = int(cxf.winner(grid, self._k))
            self._filled = int(np.count_nonzero(grid))
        return self.state, self.actions

    def reset(self, state: State | None = None) -> tuple[State, Actions]:
        return self.start(state)

    # ------------------------------------------------------------------

    def transition(self, action: Action) -> tuple[State, Actions]:
        if self._grid is None or self._info is None:
            raise RuntimeError("Call start() before transition().")
        if self.terminal():
            raise RuntimeError("Game is over; call start() before moving again.")

        index = int(action)
        if index < 0 or index >= self._cells:
            raise ValueError(f"Illegal action {action!r}: outside the action space")
        row, col = self._coords(index)
        if self._grid[row, col] != 0:
            raise ValueError(f"Illegal action {action!r}: cell ({row}, {col}) is taken")

        self._undo_stack.append(
            (self._grid, copy_info(self._info), self._winner_token, self._filled)
        )

        seat = self._info["active"]
        token = np.uint8(self._config["players"][seat])
        grid = np.array(self._grid, dtype=np.uint8)
        grid[row, col] = token
        self._grid = grid
        self._filled += 1
        self._winner_token = int(cxf.winner_at(grid, self._k, row, col))
        self._info["time"] += 1
        self._info["active"] = (seat + 1) % len(self._config["players"])

        if self._steps is not None:
            self._steps.append(
                TrajectoryStep(
                    action=index,
                    player_index=seat,
                    player_token=int(token),
                    time_after=self._info["time"],
                    reward=float(self.rewards()[seat]),
                    terminal_after=self.terminal(),
                )
            )
        return self.state, self.actions

    def step(self, action: Action) -> StepResult:
        state, actions = self.transition(action)
        return StepResult(state, actions, self.rewards(), self.terminal(), self.report())

    def undo(self) -> bool:
        if not self._undo_stack:
            return False
        self._grid, self._info, self._winner_token, self._filled = self._undo_stack.pop()
        if self._steps:
            self._steps.pop()
        return True

    # ------------------------------------------------------------------

    def terminal(self) -> bool:
        if self._grid is None:
            return False
        return self._winner_token != 0 or self._filled >= self._cells

    def winner(self) -> WinnerInfo | None:
        if self._winner_token == 0:
            return None
        return {
            "token": self._winner_token,
            "id": self._config["players"].index(self._winner_token),
        }

    def report(self) -> Report:
        steps = 0 if self._info is None else self._info["time"]
        won = self.winner()
        if won is not None:
            return {"winner": won, "steps": steps, "tie": False}
        return {"winner": None, "steps": steps, "tie": self.terminal()}

    def rewards(self) -> Rewards:
        out = np.zeros(len(self._config["players"]), dtype=np.float64)
        if not self.terminal():
            return out
        won = self.winner()
        if won is None:
            out[:] = self._rewards.draw
            return out
        out[:] = self._rewards.loss
        out[won["id"]] = self._rewards.win
        return out

    # ------------------------------------------------------------------

    def legal_actions(self) -> Actions:
        return self.actions

    @property
    def state(self) -> State:
        if self._grid is None or self._info is None:
            raise RuntimeError("Call start() before accessing state.")
        grid = self._grid.view()
        grid.setflags(write=False)
        return {"grid": grid, "info": copy_info(self._info)}

    @property
    def actions(self) -> Actions:
        if self._grid is None:
            raise RuntimeError("Call start() before accessing actions.")
        mask = (self._grid.reshape(-1) == 0).astype(np.uint8)
        mask.setflags(write=False)
        return mask

    def trajectory(self) -> Trajectory:
        if self._steps is None:
            raise RuntimeError("Recording was not enabled (set record=True).")
        return Trajectory(
            config=self._config,
            steps=list(self._steps),
            report=self.report() if self._info is not None else None,
        )
