from __future__ import annotations

import numpy as np

import connectx.functional as cxf
from connectx.config import validate_config
from connectx.renderer import terminal_render as render
from connectx.trajectory import Trajectory, TrajectoryStep
from connectx.types import (
    Action,
    Actions,
    Config,
    Grid,
    Info,
    Instance,
    Report,
    Rewards,
    RewardSpec,
    State,
    StepResult,
    WinnerInfo,
    copy_info,
)

__all__ = ["Game"]


class Game:
    """Drop Connect on a grid; implements :class:`connectx.engine.GameEngine`.

    The winner is tracked incrementally: each move only checks the four lines
    through the cell the token landed on, so :meth:`terminal` and
    :meth:`report` are ``O(1)`` after an ``O(k)`` update rather than rescanning
    the board. This is what makes self-play and tree search affordable on large
    boards.

    ``place_token`` is copy-on-write: each move returns a new grid buffer, so
    :meth:`fork` can share the current grid with a branched game until one path
    moves. Set ``undo=True`` to push snapshots before each move and rewind with
    :meth:`undo` — cheaper than forking when you are walking a search tree.
    Set ``record=True`` to append :class:`TrajectoryStep` entries after each move.
    """

    def __init__(
        self,
        config: Config,
        *,
        undo: bool = False,
        record: bool = False,
        record_grid_snapshots: bool = False,
        rewards: RewardSpec = RewardSpec(),
    ) -> None:
        self._config: Config = validate_config(config)
        self._rewards = RewardSpec(*rewards)
        self._grid: Grid | None = None
        self._actions: Actions | None = None
        self._info: Info | None = None
        self._winner_token: int = 0
        self._filled: int = 0
        self._cells: int = int(config["shape"][0]) * int(config["shape"][1])
        self._undo_stack: list[tuple[Grid, Info, int, int]] | None = [] if undo else None
        self._record = record
        self._record_grid_snapshots = record_grid_snapshots
        self._trajectory_steps: list[TrajectoryStep] | None = [] if record else None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def config(self) -> Config:
        return self._config

    @property
    def reward_spec(self) -> RewardSpec:
        return self._rewards

    def start(self, state: State | None = None) -> tuple[State, Actions]:
        """Begin a new episode, or resume from ``state``.

        Resuming rescans the board for a winner, since there is no last move to
        check incrementally.
        """
        if state is None:
            self._grid = cxf.create_grid(self._config["shape"])
            self._info = {"active": 0, "time": 0}
            self._winner_token = 0
            self._filled = 0
            if self._undo_stack is not None:
                self._undo_stack.clear()
            if self._trajectory_steps is not None:
                self._trajectory_steps.clear()
        else:
            grid = np.array(state["grid"], dtype=np.uint8)
            if grid.shape != tuple(self._config["shape"]):
                raise ValueError(
                    f"state grid has shape {grid.shape}, "
                    f"config expects {tuple(self._config['shape'])}"
                )
            n_players = len(self._config["players"])
            active = int(state["info"]["active"])
            if active < 0 or active >= n_players:
                raise ValueError(
                    f"state active index {active} out of range for {n_players} players"
                )
            self._grid = grid
            self._info = {"active": active, "time": int(state["info"]["time"])}
            self._winner_token = int(cxf.winner(self._grid, self._config["k"]))
            self._filled = int(np.count_nonzero(self._grid))

        self._actions = cxf.generate_actions(self._grid)
        return self.state, self.actions

    def reset(self, state: State | None = None) -> tuple[State, Actions]:
        """Alias for :meth:`start` (RL-style naming)."""
        return self.start(state)

    # ------------------------------------------------------------------
    # Transitions
    # ------------------------------------------------------------------

    def transition(self, action: Action) -> tuple[State, Actions]:
        """Apply one move. Raises :class:`ValueError` on an illegal action."""
        if self._grid is None or self._info is None or self._actions is None:
            raise RuntimeError("Call start() before transition().")
        if self.terminal():
            raise RuntimeError(
                "Game is over; call start() before moving again "
                f"(report: {self.report()})"
            )

        column = int(action)
        row = cxf.drop_row(self._grid, column)
        if row < 0:
            raise ValueError(
                f"Illegal action {action!r}: column is full or out of range "
                f"(legal columns: {cxf.valid_action_columns(self._actions).tolist()})"
            )

        if self._undo_stack is not None:
            self._undo_stack.append(
                (self._grid, copy_info(self._info), self._winner_token, self._filled)
            )

        player_index = self._info["active"]
        token = np.uint8(self._config["players"][player_index])

        self._grid = cxf.place_token(self._grid, token, column)
        self._filled += 1
        self._winner_token = int(
            cxf.winner_at(self._grid, self._config["k"], row, column)
        )

        self._info["time"] += 1
        self._info["active"] = self._info["time"] % len(self._config["players"])

        self._actions = cxf.generate_actions(self._grid)

        if self._trajectory_steps is not None:
            terminated = self.terminal()
            self._trajectory_steps.append(
                TrajectoryStep(
                    action=column,
                    player_index=player_index,
                    player_token=int(token),
                    time_after=self._info["time"],
                    reward=float(self.rewards()[player_index]),
                    terminal_after=terminated,
                    grid_after=(
                        np.copy(self._grid) if self._record_grid_snapshots else None
                    ),
                )
            )

        return self.state, self.actions

    def step(self, action: Action) -> StepResult:
        """Apply one move and return the RL-shaped transition.

        Unlike :meth:`transition`, this also returns per-seat rewards, the
        terminal flag, and the outcome report, so a training loop does not have
        to reconstruct them.
        """
        state, actions = self.transition(action)
        return StepResult(
            state=state,
            actions=actions,
            rewards=self.rewards(),
            terminated=self.terminal(),
            report=self.report(),
        )

    def undo(self) -> bool:
        """Rewind one move. Returns False when there is nothing to undo."""
        if not self._undo_stack:
            return False
        grid, info, winner_token, filled = self._undo_stack.pop()
        self._grid = grid
        self._info = info
        self._winner_token = winner_token
        self._filled = filled
        self._actions = cxf.generate_actions(self._grid)
        if self._trajectory_steps:
            self._trajectory_steps.pop()
        return True

    def fork(self) -> Game:
        """Branch simulation: shares the grid with this game until a move is made."""
        if self._grid is None or self._info is None:
            raise RuntimeError("Call start() before fork().")
        child = Game(
            self._config,
            undo=False,
            record=self._record,
            record_grid_snapshots=self._record_grid_snapshots,
            rewards=self._rewards,
        )
        child._grid = self._grid
        child._info = copy_info(self._info)
        child._actions = self._actions
        child._winner_token = self._winner_token
        child._filled = self._filled
        if child._trajectory_steps is not None:
            child._trajectory_steps.clear()
        return child

    # ------------------------------------------------------------------
    # Outcome
    # ------------------------------------------------------------------

    def terminal(self) -> bool:
        """Someone has a line, or the board is full."""
        if self._grid is None:
            return False
        return self._winner_token != 0 or self._filled >= self._cells

    def winner(self) -> WinnerInfo | None:
        """The winning seat, or None if nobody has won (yet or ever)."""
        if self._winner_token == 0:
            return None
        return {
            "token": self._winner_token,
            "id": self._config["players"].index(self._winner_token),
        }

    def report(self) -> Report:
        """Outcome summary. ``winner`` and ``tie`` are both empty mid-game."""
        steps = 0 if self._info is None else self._info["time"]
        won = self.winner()
        if won is not None:
            return {"winner": won, "steps": steps, "tie": False}
        return {"winner": None, "steps": steps, "tie": self.terminal()}

    def rewards(self) -> Rewards:
        """Per-seat terminal reward, aligned with ``config['players']``.

        All zeros until the game ends.
        """
        n_players = len(self._config["players"])
        out = np.zeros(n_players, dtype=np.float64)
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
    # Views
    # ------------------------------------------------------------------

    def legal_actions(self) -> Actions:
        """Read-only mask of playable columns."""
        return self.actions

    @property
    def state(self) -> State:
        """A read-only view of the board plus a copy of the turn counters.

        The grid is a non-writeable view rather than a copy: agents get a cheap
        observation they cannot use to corrupt the game. Call ``np.array(grid)``
        if you need a mutable one.
        """
        if self._grid is None or self._info is None:
            raise RuntimeError("Call start() before accessing state.")
        grid = self._grid.view()
        grid.setflags(write=False)
        return {"grid": grid, "info": copy_info(self._info)}

    @property
    def actions(self) -> Actions:
        if self._actions is None:
            raise RuntimeError("Call start() before accessing actions.")
        actions = self._actions.view()
        actions.setflags(write=False)
        return actions

    @property
    def action_space_size(self) -> int:
        """One action per column: tokens are dropped, not placed freely."""
        return int(self._config["shape"][1])

    @property
    def active_player(self) -> int:
        """Token of the player to move."""
        if self._info is None:
            raise RuntimeError("Call start() before accessing active_player.")
        return int(self._config["players"][self._info["active"]])

    def instance(self) -> Instance:
        """Bundle the current position with its config, ready for `utils.save`."""
        state = self.state
        return {
            "grid": np.array(state["grid"], dtype=np.uint8),
            "info": state["info"],
            "config": self._config,
        }

    @classmethod
    def from_instance(cls, instance: Instance, **kwargs) -> Game:
        """Rebuild a game positioned at a saved instance."""
        game = cls(instance["config"], **kwargs)
        game.start({"grid": instance["grid"], "info": instance["info"]})
        return game

    def trajectory(self) -> Trajectory:
        """The episode recorded so far. Requires ``record=True``."""
        if self._trajectory_steps is None:
            raise RuntimeError("Recording was not enabled (set record=True).")
        return Trajectory(
            config=self._config,
            steps=list(self._trajectory_steps),
            report=self.report() if self._info is not None else None,
        )

    def render(self) -> None:
        if self._grid is None or self._info is None:
            raise RuntimeError("Call start() before render().")
        render(
            self._grid,
            self._info["time"],
            self._config["players"][self._info["active"]],
            k=self._config["k"],
        )

    def __repr__(self) -> str:
        rows, cols = self._config["shape"]
        started = self._grid is not None
        time = self._info["time"] if self._info is not None else 0
        return (
            f"Game(shape=({rows}, {cols}), k={self._config['k']}, "
            f"players={self._config['players']}, "
            f"{'time=' + str(time) if started else 'not started'})"
        )
