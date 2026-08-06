from connectx.types import Config, State
from connectx.game import Game

from agents.types import Agent
from agents import RandomAgent


def run(config: Config, agents: list[Agent]) -> tuple[State, dict]:
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

    return state, game.report()


if __name__ == "__main__":
    config: Config = {"shape": (6, 7), "k": 4, "players": [1, 2]}

    agents = [RandomAgent(), RandomAgent()]

    final_state, report = run(config, agents)
    print("Final state:", final_state)
    print("Report:", report)
