"""Keyboard agent, so a person can sit in any seat of any variant."""

from __future__ import annotations

import connectx.functional as cxf
from agents.types import BaseAgent
from connectx.types import Action, Actions, State

__all__ = ["HumanAgent"]


class HumanAgent(BaseAgent):
    def select(self, state: State, actions: Actions) -> Action:
        legal = [int(c) for c in cxf.valid_action_columns(actions)]
        if not legal:
            raise ValueError("no legal actions available")
        prompt = f"column {legal} > "
        while True:
            try:
                raw = input(prompt)
            except EOFError as exc:
                raise KeyboardInterrupt("input closed") from exc
            raw = raw.strip()
            if raw in {"q", "quit", "exit"}:
                raise KeyboardInterrupt("player quit")
            try:
                column = int(raw)
            except ValueError:
                print(f"  '{raw}' is not a column number")
                continue
            if column not in legal:
                print(f"  column {column} is not playable")
                continue
            return column
