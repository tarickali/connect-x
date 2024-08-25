from typing import Protocol

from connectx.types import State, Action, Actions


class Agent(Protocol):
    def select(self, state: State, actions: Actions) -> Action: ...


# class Actor(Protocol):
#     def act(self, state: State, actions: Actions) -> Action: ...


# class Agent(Selector, Actor, Protocol): ...
