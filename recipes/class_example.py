"""Class API: run a game to completion and print the outcome.

Run with ``python recipes/class_example.py``.
"""

from agents import RandomAgent, agent_reset
from agents.types import Agent
from connectx import Game, RewardSpec, StepResult, preset
from connectx.types import Config


def run(config: Config, agents: list[Agent], render: bool = True) -> StepResult:
    game = Game(config, rewards=RewardSpec(win=1.0, loss=-1.0, draw=0.0))
    state, actions = game.start()
    for agent in agents:
        agent_reset(agent, config)

    result = StepResult(state, actions, game.rewards(), False, game.report())
    while not game.terminal():
        if render:
            game.render()
        action = agents[state["info"]["active"]].select(state, actions)
        # step() returns the RL-shaped transition: rewards and done included.
        result = game.step(action)
        state, actions = result.state, result.actions

    if render:
        game.render()
    return result


if __name__ == "__main__":
    config = preset("connect4")
    agents = [RandomAgent(seed=0), RandomAgent(seed=1)]

    result = run(config, agents)
    print("Report :", result.report)
    print("Rewards:", result.rewards.tolist())
