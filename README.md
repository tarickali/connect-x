# connect-x

A general implementation of the Connect-based games to train and test AI agents.

## Overview

In order to push the boundaries of AI research in the pursuit of creating more intelligently capable agents, we need rich and complex environments to train and evaluate them. Although there already exist many research environments that provide interesting and difficult tasks from a wide range of domains, there are only a few projects that provide environments cater made to test the ability of an agent to generalize across task domains. This project is meant to help fill that gap.

Although Connect 4 is a relatively simple game that even children can learn and master, it nevertheless provides a rich and intellectual stimulating environment to train and evaluate agents on. However, since we wish to evaluate the ability of agents to generalize across task domains, having agents train only on Connect 4 would not be sufficient. Instead, we extend the game of Connect 4 to a more general version known as `connectx`, where one can configure the game parameters to quickly make different games.

This creates a universe of environments that have fundamentally similarities and can serve as a stepping stone to build generally capable game-playing agents.

## Features

- Customizable games using config objects.
- Optimized execution using `numba`.
- Utility functions to monitor and evaluate games.

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

The base version of this project provides three configuration options to define a game:

1. `shape: tuple[int, int]`: The shape of the game board. Requirement: `shape[0] > 0, shape[1] > 0`.
2. `k: int`: The length of the line to win the game. Requirement: `k < max(shape[0], shape[1])`.
3. `players: list[int]`: The number and identity of each player. Requirement: `len(players) > 1` and that there are at least two unique values in `players`.

These are used to define a simple game, however, one can extend these to allow for more interesting variants.

## Extensions

At its current state, this project only provides an interface to create parameterized variants of Connect 4. That is, essentially the game of Connect 4 with a different sized boards and line lengths to win the game. However, Connect 4 is part of a family of games known as m,n,k-game where within this family that are different games with different transition and termination rules.

To extend this project in a meaningful way, one can use the primitives provided by this project directly or use them as inspiration to construct different types of games entirely. For example, the game of Gomuko, Connect6, Pente. Furthermore, there are even variants within Connect 4 itself with different game rules such as Pop 10, Pop Out, and Power Up.

These provide rich environments to train agents on as well as good practice to design and build configurable versions of each new game.

## Contributions

Contributions are always welcome! If you have any suggestions on ways to improve or extend this project please clone the repo, implement the changes, and create a pull request.

If you would like to reach out to me to discuss your ideas or this project's mission in general feel free to reach out to me.

## License

This project is under the Apache License Version 2.0. For full details please refer to the [license file](LICENSE).
