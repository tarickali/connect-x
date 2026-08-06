from __future__ import annotations

from typing import List, Tuple, Optional

import numpy as np

from connectx.types import Action, Actions, Config, Grid, Info, Report, State
import connectx.functional as cxf
from connectx.renderer import terminal_render as render
from connectx.trajectory import Trajectory, TrajectoryStep


class Game:
    """Drop Connect on a grid; implements `connectx.engine.GameEngine`.

    `place_token` uses copy-on-write: each move returns a new grid buffer, while
    `fork` shares the current grid with a branched game until one path moves.

    Set `undo=True` to push snapshots before each move; use `undo` to pop.
    Set `record=True` to append `TrajectoryStep` entries after each move.
    """

    def __init__(
        self,
        config: Config,
        *,
        undo: bool = False,
        record: bool = False,
        record_grid_snapshots: bool = False,
    ) -> None:
        self._config: Config = config
        self._grid: Optional[Grid] = None
        self._actions: Optional[Actions] = None
        self._info: Optional[Info] = None
        self._undo_stack: Optional[List[Tuple[Grid, Info]]] = [] if undo else None
        self._record = record
        self._record_grid_snapshots = record_grid_snapshots
        self._trajectory_steps: Optional[list[TrajectoryStep]] = [] if record else None

    @property
    def config(self) -> Config:
        return self._config

    def start(self, state: Optional[State] = None) -> tuple[State, Actions]:
        if state is None:
            self._grid = cxf.create_grid(self._config["shape"])
            self._info = {"active": 0, "time": 0}
            if self._undo_stack is not None:
                self._undo_stack.clear()
            if self._trajectory_steps is not None:
                self._trajectory_steps.clear()
        else:
            self._grid = state["grid"]
            self._info = dict(state["info"])

        self._actions = cxf.generate_actions(self._grid)

        return self.state, self.actions

    def reset(self, state: Optional[State] = None) -> tuple[State, Actions]:
        return self.start(state)

    def step(self, action: Action) -> tuple[State, Actions]:
        return self.transition(action)

    def transition(self, action: Action) -> tuple[State, Actions]:
        if self._grid is None or self._info is None or self._actions is None:
            raise RuntimeError("Call start() before transition().")

        a = int(action)
        if a < 0 or a >= self._actions.shape[0] or self._actions[a] != 1:
            raise ValueError(f"Illegal action {action!r}")

        if self._undo_stack is not None:
            self._undo_stack.append((np.copy(self._grid), dict(self._info)))

        player_index = self._info["active"]
        token = np.uint8(self._config["players"][player_index])

        self._grid = cxf.place_token(self._grid, token, np.uint8(a))

        self._info["time"] += 1
        self._info["active"] = self._info["time"] % len(self._config["players"])

        self._actions = cxf.generate_actions(self._grid)

        if self._trajectory_steps is not None:
            self._trajectory_steps.append(
                TrajectoryStep(
                    action=a,
                    player_index=player_index,
                    player_token=int(token),
                    time_after=self._info["time"],
                    terminal_after=self.terminal(),
                    grid_after=(
                        np.copy(self._grid) if self._record_grid_snapshots else None
                    ),
                )
            )

        return self.state, self.actions

    def undo(self) -> bool:
        if not self._undo_stack:
            return False
        grid, info = self._undo_stack.pop()
        self._grid = grid
        self._info = info
        self._actions = cxf.generate_actions(self._grid)
        if self._trajectory_steps:
            self._trajectory_steps.pop()
        return True

    def fork(self) -> Game:
        """Branch simulation: shares the grid with this game until a move is made."""
        child = Game(
            self._config,
            undo=False,
            record=self._record,
            record_grid_snapshots=self._record_grid_snapshots,
        )
        if self._grid is None or self._info is None:
            raise RuntimeError("Call start() before fork().")
        child._grid = self._grid
        child._info = {"active": self._info["active"], "time": self._info["time"]}
        child._actions = self._actions
        if child._trajectory_steps is not None:
            child._trajectory_steps.clear()
        return child

    def trajectory(self) -> Trajectory:
        if self._trajectory_steps is None:
            raise RuntimeError("Recording was not enabled (set record=True).")
        return Trajectory(config=self._config, steps=list(self._trajectory_steps))

    def terminal(self) -> bool:
        if self._grid is None:
            return False
        return cxf.terminal(self._grid, self._config["k"])

    def render(self) -> None:
        if self._grid is None or self._info is None:
            raise RuntimeError("Call start() before render().")
        render(
            self._grid,
            self._info["time"],
            self._config["players"][self._info["active"]],
        )

    def report(self) -> Report:
        if self._info is None:
            return {"winner": None, "steps": 0, "tie": False}

        steps = self._info["time"]

        # Not terminal yet
        if not self.terminal():
            return {
                "winner": None,
                "steps": steps,
                "tie": False,
            }

        if self._grid is None:
            return {"winner": None, "steps": steps, "tie": False}

        is_tie = cxf.check_tie(self._grid)
        if is_tie:
            return {
                "winner": None,
                "steps": steps,
                "tie": True,
            }

        winner_id = (steps - 1) % len(self._config["players"])
        winner_token = self._config["players"][winner_id]

        return {
            "winner": {"token": winner_token, "id": winner_id},
            "steps": steps,
            "tie": False,
        }

    def legal_actions(self) -> Actions:
        if self._actions is None:
            raise RuntimeError("Call start() before legal_actions().")
        return self._actions

    @property
    def state(self) -> State:
        if self._grid is None or self._info is None:
            raise RuntimeError("Call start() before accessing state.")
        return {"grid": self._grid, "info": self._info}

    @property
    def actions(self) -> Actions:
        if self._actions is None:
            raise RuntimeError("Call start() before accessing actions.")
        return self._actions
