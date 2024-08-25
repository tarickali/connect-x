# connect-x

> A general implementation of the classic game Connect 4 to train and test AI agents.

## Overview

In order to push the boundaries of AI research in the pursuit of creating more intelligently capable agents, we need rich and complex environments to train and evaluate them. Although there already exist many research environments that provide interesting and difficult tasks from a wide range of domains, there are only a few projects that provide environments cater made to test the ability of an agent to generalize across task domains. This project is meant to help fill that gap.

Although Connect 4 is a relatively simple game that even children can learn and master, it nevertheless provides a rich and intellectual stimulating environment to train and evaluate agents on. However, since we wish to evaluate the ability of agents to generalize across task domains, having agents train only on Connect 4 would not be sufficient. Instead, we extend the game of Connect 4 to a more general version known as `connectx`, where one can configure the game parameters to quickly make different games.

This creates a universe of environments that have fundamentally similarities.

## Features

- Customizable games using config objects.
- Optimized execution using `numba`.
- Saving and loading games

## Installation

All requirements can be found in [requirements.txt](./requirements.txt). To install this project, follow the following steps:

1. Clone repo

```{bash}
git clone https://github.com/tarickali/connectx.git
```

2. Create a virtual environment and install project requirements within the cloned repo

```{bash}
cd connectx

# Option 1. Using venv
python -m venv connectx
source connectx/bin/activate

# Option 2. Using conda
conda create -n connectx
conda activate connectx

# Install pip modules
pip install -r requirements.txt
```

And that's it!

## Usage

There are two ways to interface with `connectx`: through a functional API and a class API. Below are examples of both APIs.

**Functional**:

```{python3}
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
```

**Class**:

```{python3}
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
```

More examples on how to use the project can be found under the `/recipes` directory.

## Configuration Options

## Extensions

## Contributions

## License
