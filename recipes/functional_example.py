from connectx.types import Config, State
import connectx.functional as cxf
from connectx.utils import make_state
from connectx.renderer import terminal_render as render

from agents.types import Agent
from agents import RandomAgent


def run(config: Config, agents: list[Agent]) -> State:
    shape, k, players = config["shape"], config["k"], config["players"]

    # Create the state
    grid = cxf.create_grid(shape)
    time = 0
    active = 0

    # Create the actions
    actions = cxf.generate_actions(grid)

    while not cxf.terminal(grid, k):
        # Render the current state
        render(grid, time, players[active])
        # Select action
        action = agents[active].select(make_state(grid, time, active), actions)
        print(f"Action: {action}")
        # Execute action and update state
        grid = cxf.place_token(grid, players[active], action)
        time += 1
        active = time % 2
        # Generate valid actions
        actions = cxf.generate_actions(grid)

    return {"grid": grid, "active": active, "time": time}


if __name__ == "__main__":
    config: Config = Config(shape=(6, 7), k=4, players=[1, 2])

    agents = [RandomAgent(), RandomAgent()]

    report = run(config, agents)
    print(report)
