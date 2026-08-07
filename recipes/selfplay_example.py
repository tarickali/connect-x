"""Self-play: collect episodes into a ReplayMemory and build training tensors.

This is the shape a learner consumes — encoded planes, an action mask, a policy
target, and a value target — without any learning code in the loop yet.

Run with ``python recipes/selfplay_example.py``.
"""

import numpy as np

from agents import agent_reset, make_agent
from connectx import Game, ReplayMemory, preset
from connectx.encoding import encode_batch, encode_state
from connectx.types import Config


def collect(config: Config, episodes: int = 25, seed: int = 0) -> ReplayMemory:
    memory = ReplayMemory(max_episodes=episodes)
    game = Game(config, record=True)

    for episode in range(episodes):
        # Reseed per episode so a run is reproducible from a single seed.
        agents = [
            make_agent("mcts:simulations=64", config, seed=seed + episode * 10 + i)
            for i in range(len(config["players"]))
        ]
        for agent in agents:
            agent_reset(agent, config)

        state, actions = game.start()
        while not game.terminal():
            action = agents[state["info"]["active"]].select(state, actions)
            state, actions = game.transition(action)
        memory.push(game.trajectory())

    return memory


def build_dataset(memory: ReplayMemory) -> dict[str, np.ndarray]:
    """Flatten episodes into per-position training arrays."""
    observations, policies, values = [], [], []
    for trajectory in memory:
        returns = trajectory.returns()
        n_columns = trajectory.config["shape"][1]
        for index, (state, action) in enumerate(trajectory.replay()):
            observations.append(encode_state(state, trajectory.config))
            target = np.zeros(n_columns, dtype=np.float32)
            target[action] = 1.0
            policies.append(target)
            values.append(returns[index])
    return {
        "observations": np.stack(observations) if observations else np.zeros((0,)),
        "policies": np.stack(policies) if policies else np.zeros((0,)),
        "values": np.array(values, dtype=np.float32),
    }


if __name__ == "__main__":
    config = preset("small")
    memory = collect(config, episodes=20, seed=0)
    print(f"episodes: {len(memory)}  transitions: {memory.steps}")

    data = build_dataset(memory)
    for name, array in data.items():
        print(f"  {name:13} {array.shape} {array.dtype}")

    # Mirror symmetry doubles the data for free on Connect-style boards.
    augmented = encode_batch(data["observations"], mirror=True)
    print(f"  with mirror   {augmented.shape}")
