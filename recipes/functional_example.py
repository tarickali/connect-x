"""Functional API: hold the grid yourself and call the JIT primitives.

Best for custom training loops and numba-heavy pipelines, where you do not want
an object owning the state. Note the explicit ``is_legal`` guard: unlike
:class:`connectx.Game`, the functional layer will not raise on an illegal
column, so the loop has to check before it advances the clock.

Run with ``python recipes/functional_example.py``.
"""

import numpy as np

import connectx.functional as cxf
from agents import RandomAgent, agent_reset
from agents.types import Agent
from connectx import preset
from connectx.renderer import terminal_render as render
from connectx.types import Config, State
from connectx.utils import make_state


def run(config: Config, agents: list[Agent], render_board: bool = True) -> State:
    shape, k, players = config["shape"], config["k"], config["players"]
    for agent in agents:
        agent_reset(agent, config)

    grid = cxf.create_grid(shape)
    time, active = 0, 0
    actions = cxf.generate_actions(grid)

    while not cxf.terminal(grid, k):
        if render_board:
            render(grid, time, players[active], k=k, players=players)

        action = agents[active].select(make_state(grid, time, active), actions)
        if not cxf.is_legal(grid, action):
            raise ValueError(f"agent {active} chose illegal column {action}")

        grid = cxf.place_token(grid, np.uint8(players[active]), action)
        time += 1
        active = time % len(players)
        actions = cxf.generate_actions(grid)

    if render_board:
        render(grid, time, players[active], k=k, players=players)
    return make_state(grid, time, active)


if __name__ == "__main__":
    config = preset("connect4")
    agents = [RandomAgent(seed=0), RandomAgent(seed=1)]

    final_state = run(config, agents)
    token = int(cxf.winner(final_state["grid"], config["k"]))
    print("Winner token:", token or "draw")
