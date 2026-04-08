import sys
from pathlib import Path

# Allow `python scripts/minimax_random_experiment.py` from repo root (script dir is on sys.path, not the package root).
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agents import MinimaxAgent, RandomAgent
from connectx import Game
from connectx.types import Config


def run():
    config: Config = {"shape": (6, 7), "k": 4, "players": [1, 2]}
    agents = [MinimaxAgent(config, depth=3), RandomAgent()]
    game = Game(config)
    game.start()

    while not game.terminal():
        game.render()
        action = agents[game.state["info"]["active"]].select(game.state, game.actions)
        print(f"Action: {action}")
        game.transition(action)
    game.render()

    print(game.report())


if __name__ == "__main__":
    run()
