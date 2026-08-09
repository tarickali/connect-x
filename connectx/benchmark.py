"""Throughput benchmarks for the engine and the agent ladder.

Reported as moves/second rather than seconds/call, because the number that
matters for RL is how many environment transitions a training run can afford
per hour.

All timings exclude numba's first-call compilation: every function is warmed up
before it is measured, so the numbers reflect steady state.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

import connectx.functional as cxf
from connectx.config import make_config
from connectx.game import Game
from connectx.types import Config
from connectx.utils import make_state
from connectx.vector import VecGame

__all__ = ["time_call", "run_benchmarks"]

_SHAPES = [(6, 7), (8, 9), (12, 14), (20, 20), (60, 60)]


def time_call(function: Callable[..., Any], *args, calls: int = 20_000) -> float:
    """Mean microseconds per call, after a warm-up call."""
    function(*args)
    started = time.perf_counter()
    for _ in range(calls):
        function(*args)
    return (time.perf_counter() - started) / calls * 1e6


def _bench_functional(calls: int) -> list[dict[str, Any]]:
    rows = []
    for shape in _SHAPES:
        grid = cxf.create_grid(shape)
        played = cxf.place_token(grid, np.uint8(1), shape[1] // 2)
        rows.append(
            {
                "shape": list(shape),
                "generate_actions_us": time_call(cxf.generate_actions, grid, calls=calls),
                "place_token_us": time_call(
                    cxf.place_token, grid, np.uint8(1), shape[1] // 2, calls=calls
                ),
                "winner_at_us": time_call(
                    cxf.winner_at, played, 4, shape[0] - 1, shape[1] // 2, calls=calls
                ),
                "winner_scan_us": time_call(cxf.winner, played, 4, calls=calls),
            }
        )
    return rows


def _bench_game(calls: int, engine: Any = Game) -> list[dict[str, Any]]:
    rows = []
    for shape in _SHAPES:
        config = make_config(shape, 4, [1, 2])
        game = engine(config)
        game.start()

        # Warm up, then measure a stream of legal moves with resets on terminal.
        game.transition(0)
        game.start()
        moves = 0
        started = time.perf_counter()
        while moves < calls:
            legal = cxf.valid_action_columns(game.actions)
            if game.terminal() or legal.size == 0:
                game.start()
                continue
            game.transition(int(legal[moves % legal.size]))
            moves += 1
        elapsed = time.perf_counter() - started
        rows.append(
            {
                "shape": list(shape),
                "transition_us": elapsed / moves * 1e6,
                "moves_per_second": moves / elapsed,
            }
        )
    return rows


def _bench_vector(config: Config, calls: int) -> list[dict[str, Any]]:
    rows = []
    rng = np.random.default_rng(0)
    for num_envs in (1, 16, 256, 2048):
        vec = VecGame(config, num_envs)
        vec.reset()
        vec.step(vec.sample_actions(rng))  # warm up the kernel

        steps = max(calls // max(num_envs, 1), 20)
        started = time.perf_counter()
        for _ in range(steps):
            vec.step(vec.sample_actions(rng))
        elapsed = time.perf_counter() - started
        rows.append(
            {
                "num_envs": num_envs,
                "batched_steps": steps,
                "moves_per_second": steps * num_envs / elapsed,
            }
        )
    return rows


def _bench_agents(config: Config, specs: list[str]) -> list[dict[str, Any]]:
    from agents import make_agent

    rows = []
    grid = cxf.create_grid(config["shape"])
    state = make_state(grid, time=0, active=0)
    actions = cxf.generate_actions(grid)
    for spec in specs:
        agent = make_agent(spec, config, seed=0)
        agent.select(state, actions)  # warm up numba and the JIT paths

        # Reset before every timed move: a warm transposition table would make
        # repeat searches of the same position look thousands of times faster
        # than the first one an agent actually performs in a game.
        elapsed = 0.0
        repeats = 0
        while elapsed < 0.25 and repeats < 2000:
            agent.reset(config)
            started = time.perf_counter()
            agent.select(state, actions)
            elapsed += time.perf_counter() - started
            repeats += 1
        rows.append(
            {
                "agent": spec,
                "seconds_per_move": elapsed / repeats,
                "nodes": getattr(agent, "nodes", None),
            }
        )
    return rows


def run_benchmarks(
    config: Config | None = None,
    *,
    quick: bool = False,
    json_path: str | None = None,
) -> dict[str, Any]:
    """Run every benchmark and print a table. Returns the raw results."""
    config = config or make_config((6, 7), 4, [1, 2])
    calls = 2_000 if quick else 20_000

    print("connect-x benchmarks")
    print(
        f"variant: {tuple(config['shape'])} k={config['k']} "
        f"players={list(config['players'])}\n"
    )

    functional = _bench_functional(calls)
    print("functional primitives (microseconds per call)")
    print(
        f"  {'shape':<10} {'actions':>9} {'place':>9} {'winner_at':>11} {'full scan':>11}"
    )
    for row in functional:
        print(
            f"  {str(tuple(row['shape'])):<10} {row['generate_actions_us']:9.3f} "
            f"{row['place_token_us']:9.3f} {row['winner_at_us']:11.3f} "
            f"{row['winner_scan_us']:11.3f}"
        )

    game = _bench_game(calls)
    print("\nGame.transition")
    print(f"  {'shape':<10} {'us/move':>9} {'moves/s':>12}")
    for row in game:
        print(
            f"  {str(tuple(row['shape'])):<10} {row['transition_us']:9.3f} "
            f"{row['moves_per_second']:12,.0f}"
        )

    vector = _bench_vector(config, calls)
    print("\nVecGame (batched)")
    print(f"  {'envs':>6} {'moves/s':>14}")
    for row in vector:
        print(f"  {row['num_envs']:6d} {row['moves_per_second']:14,.0f}")

    specs = ["random", "greedy", "minimax:depth=4", "mcts:simulations=200"]
    if quick:
        specs = ["random", "greedy", "minimax:depth=4"]
    agent_rows = _bench_agents(config, specs)
    print("\nagent cost for one opening move (cold search state)")
    print(f"  {'agent':<24} {'ms/move':>10} {'moves/s':>10} {'nodes':>12}")
    for row in agent_rows:
        nodes = "-" if row["nodes"] is None else f"{row['nodes']:,}"
        seconds = row["seconds_per_move"]
        print(
            f"  {row['agent']:<24} {seconds * 1e3:10.3f} "
            f"{1.0 / seconds if seconds else float('inf'):10,.0f} {nodes:>12}"
        )

    results = {
        "config": {
            "shape": list(config["shape"]),
            "k": config["k"],
            "players": list(config["players"]),
        },
        "functional": functional,
        "game": game,
        "vector": vector,
        "agents": agent_rows,
    }
    if json_path:
        path = Path(json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(results, indent=2))
        print(f"\nwrote {path}")
    return results


if __name__ == "__main__":  # pragma: no cover
    run_benchmarks()
