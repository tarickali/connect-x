from connectx.types import Config, State
from connectx.game import Game

from agents.types import Agent
from agents import RandomAgent


def run(config: Config, agents: list[Agent]) -> State:
    game = Game(config)
    state, actions = game.start()

    while not game.terminal():
        # Render the current state
        game.render()
        # Select action
        action = agents[state["info"]["active"]].select(state, actions)
        print(f"Action: {action}")
        # Execute action and update state
        state, actions = game.transition(action)

    return state


if __name__ == "__main__":
    config: Config = Config(shape=(6, 7), k=4, players=[1, 2])

    agents = [RandomAgent(), RandomAgent()]

    report = run(config, agents)
    print(report)
