# connect-x

A **general, parameterized implementation of Connect-based games** for training and evaluating AI agents. Built for research and experimentation in agent generalization across task variants.

## Overview

In order to push the boundaries of AI research in the pursuit of creating more intelligently capable agents, we need rich and complex environments to train and evaluate them. Although there already exist many research environments that provide interesting and difficult tasks from a wide range of domains, there are only a few projects that provide environments cater made to test the ability of an agent to generalize across task domains. This project is meant to help fill that gap.

Although Connect 4 is a relatively simple game that even children can learn and master, it nevertheless provides a rich and intellectually stimulating environment to train and evaluate agents on. However, since we wish to evaluate the ability of agents to generalize across task domains, having agents train only on Connect 4 would not be sufficient. Instead, we extend the game of Connect 4 to a more general version known as `connectx`, where one can configure the game parameters to quickly make different games.

This creates a universe of environments that have fundamentally similarities and can serve as a stepping stone to build generally capable game-playing agents.

## Features

- **Parameterized games** — Board shape, win length `k`, and player set via a single config.
- **Two APIs** — Functional (pure, numpy/numba-friendly) and class-based `Game` for different workflows.
- **Numba-optimized core** — Grid operations and win detection compiled for speed.
- **Pluggable agents** — Simple `Agent` protocol; ship a random agent and add your own.
- **Utilities** — State serialization, terminal rendering, and helpers for training loops.

## Architecture

The project is built in three layers:

1. **Config & types** (`connectx.types`) — Game parameters (board shape, win length `k`, player IDs) and shared types for grids, state, and actions. All APIs consume and produce these types.

2. **Core engine** — Two ways to run the game:
   - **Functional** (`connectx.functional`): Pure, JIT-compiled functions — `create_grid`, `place_token`, `generate_actions`, `terminal`. No hidden state; you hold the grid and call these in a loop. Best for custom training loops and numba-heavy pipelines.
   - **Class-based** (`connectx.game.Game`): Stateful wrapper that holds grid, current player, and time. You call `start()`, then `transition(action)` until `terminal()`. Built on top of the same functional primitives.

3. **Agents** (`agents`) — Implement the `Agent` protocol (`select(state, actions) -> action`). The engine never imports agent logic; you pass in a list of agents and the runner asks the current agent for a move. That keeps the environment agnostic to how decisions are made (random, heuristic, or learned).

```
Config ──► [ functional (create_grid, place_token, generate_actions, terminal) ]
     ──► [ Game (start, transition, terminal) ] ──► Agent.select(state, actions) ──► action
```

## Installation

Requires **Python 3.10+**. Requirements are in [requirements.txt](./requirements.txt); you can also install the project (and optional test deps) with `pip install -e ".[test]"` from the repo root. Use a virtual environment.

1. **Clone the repository**

```bash
git clone https://github.com/tarickali/connect-x.git
cd connect-x
```

2. **Create a virtual environment and install dependencies**

```bash
# Option 1: venv
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

# Option 2: conda
conda create -n connectx python=3.10
conda activate connectx

# Install dependencies
pip install -r requirements.txt
```

To run tests: `pip install -e ".[test]"` then `pytest`.

## Quick Start

Run two random agents against each other (class API):

```bash
python main.py
```

Or use the recipes:

```bash
python recipes/class_example.py
python recipes/functional_example.py
```

To run a **benchmark** (generate_actions and transition timings for various board dimensions):

```bash
python scripts/benchmark.py
```

## Usage

You can use either a **functional API** (good for custom training loops and numba) or a **class-based API** (good for scripting and encapsulation).

### Class-based API

```python
from connectx import Config, Game
from connectx.types import State
from agents import RandomAgent
from agents.types import Agent


def run(config: Config, agents: list[Agent]) -> State:
    game = Game(config)
    state, actions = game.start()

    while not game.terminal():
        game.render()
        action = agents[state["info"]["active"]].select(state, actions)
        print(f"Action: {action}")
        state, actions = game.transition(action)

    return state


if __name__ == "__main__":
    config: Config = {"shape": (6, 7), "k": 4, "players": [1, 2]}
    agents = [RandomAgent(), RandomAgent()]
    final_state = run(config, agents)
    print(final_state)
```

### Functional API

```python
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
    time, active = 0, 0
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
        active = time % len(players)
        # Generate valid actions
        actions = cxf.generate_actions(grid)

    return {"grid": grid, "info": {"active": active, "time": time}}
```

Provide one agent per player (same length as `config["players"]`). **Note:** `MinimaxAgent` supports only two-player games for now; use `RandomAgent` or custom agents for more players.

More examples are in the [recipes](./recipes) directory.

## Configuration

Games are defined by a config with three fields:

| Option    | Type           | Description                                    |
| --------- | -------------- | ---------------------------------------------- |
| `shape`   | `(rows, cols)` | Board dimensions; both must be &gt; 0.         |
| `k`       | `int`          | Line length to win; must be ≤ max(rows, cols). |
| `players` | `list[int]`    | Player IDs; at least two distinct values.      |

Example: classic Connect 4 is `shape=(6, 7)`, `k=4`, `players=[1, 2]`.

## Extensions

At its current state, this project only provides an interface to create parameterized variants of Connect 4. That is, essentially the game of Connect 4 with a different sized boards and line lengths to win the game. However, Connect 4 is part of a family of games known as m,n,k-game where within this family that are different games with different transition and termination rules.

To extend this project in a meaningful way, one can use the primitives provided by this project directly or use them as inspiration to construct different types of games entirely. For example, the game of Gomuko, Connect6, Pente. Furthermore, there are even variants within Connect 4 itself with different game rules such as Pop 10, Pop Out, and Power Up.

These provide rich environments to train agents on as well as good practice to design and build configurable versions of each new game.

## Contributions

Contributions are always welcome! If you have any suggestions on ways to improve or extend this project please clone the repo, implement the changes, and create a pull request.

If you would like to reach out to me to discuss your ideas or this project's mission in general feel free to reach out to me.

## License

This project is under the Apache License Version 2.0. For full details please refer to the [license file](LICENSE).
