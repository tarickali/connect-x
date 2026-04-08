from typing import Optional, Any

from connectx.types import Config, Grid, Info, Action, Actions, State
import connectx.functional as cxf
from connectx.renderer import terminal_render as render


class Game:
    def __init__(self, config: Config) -> None:
        self._config: Config = config
        self._grid: Grid = None
        self._actions: Actions = None
        self._info: Info = None

    def start(self, state: Optional[State] = None) -> tuple[State, Actions]:
        if state is None:
            self._grid = cxf.create_grid(self._config["shape"])
            self._info = {"active": 0, "time": 0}
        else:
            self._grid = state["grid"]
            self._info = state["info"]

        self._actions = cxf.generate_actions(self._grid)

        return self.state, self.actions

    def transition(self, action: Action) -> tuple[State, Actions]:
        token = self._config["players"][self._info["active"]]
        self._grid = cxf.place_token(self._grid, token, action)

        self._info["time"] += 1
        self._info["active"] = self._info["time"] % len(self._config["players"])

        self._actions = cxf.generate_actions(self._grid)

        return self.state, self.actions

    def terminal(self) -> bool:
        return cxf.terminal(self._grid, self._config["k"])

    def render(self) -> None:
        render(
            self._grid,
            self._info["time"],
            self._config["players"][self._info["active"]],
        )

    def report(self) -> dict[str, Any]:
        steps = self._info["time"]

        # Not terminal yet
        if not self.terminal():
            return {
                "winner": None,
                "steps": steps,
                "tie": False,
            }

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

    @property
    def state(self) -> State:
        return {"grid": self._grid, "info": self._info}

    @property
    def actions(self) -> Actions:
        return self._actions
