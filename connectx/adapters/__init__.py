"""Adapters onto the standard RL APIs.

The engine deliberately does not depend on Gymnasium or PettingZoo — it stays
a plain library — but almost every training stack speaks one of them. These
modules are the bridge, and their dependencies live in the ``rl`` extra::

    pip install -e ".[rl]"

Import the submodule you need rather than pulling both in:

- :mod:`connectx.adapters.pettingzoo_env` — AEC env, the right shape for
  turn-based multi-agent play and self-play.
- :mod:`connectx.adapters.gym_env` — single-agent env where a fixed opponent
  fills the other seats, the right shape for off-the-shelf single-agent
  algorithms.
"""

__all__ = ["make_aec_env", "make_gym_env"]


def make_aec_env(*args, **kwargs):
    """Construct the PettingZoo AEC environment (imports on first use)."""
    from connectx.adapters.pettingzoo_env import ConnectXAECEnv

    return ConnectXAECEnv(*args, **kwargs)


def make_gym_env(*args, **kwargs):
    """Construct the Gymnasium single-agent environment (imports on first use)."""
    from connectx.adapters.gym_env import ConnectXGymEnv

    return ConnectXGymEnv(*args, **kwargs)
