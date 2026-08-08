"""Does numba earn its place in this project?

numba is the only non-trivial dependency here, and it is not free: it adds
~155 MB of wheels, ~1.6 s to a cold import, and couples the project to a
trailing range of numpy versions. This script measures what it buys, so the
dependency is a decision with evidence behind it rather than a habit.

Three implementations of the same primitives are compared:

  numba   the shipped code, JIT compiled
  python  the identical source with NUMBA_DISABLE_JIT=1, i.e. what you get by
          deleting the decorators
  numpy   an idiomatic vectorized implementation, i.e. the realistic
          alternative someone would write instead

The third column is the one that matters. "numba vs naive Python loops" flatters
numba; "numba vs numpy done properly" is the real question.

    python scripts/ablation.py                    # numba enabled
    NUMBA_DISABLE_JIT=1 python scripts/ablation.py    # same code, no JIT

Note that the whole test suite passes under NUMBA_DISABLE_JIT=1: numba is a
pure accelerator here, never load-bearing for behaviour.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import connectx.functional as cxf  # noqa: E402
from connectx import Game, make_config, preset  # noqa: E402
from connectx.vector import VecGame  # noqa: E402

JIT_ENABLED = os.environ.get("NUMBA_DISABLE_JIT") != "1"
LABEL = "numba" if JIT_ENABLED else "python"

SHAPES = [(6, 7), (12, 14), (30, 30), (60, 60)]


def bench(fn, *args, budget: float = 0.3) -> float:
    """Mean microseconds per call, after a warm-up."""
    fn(*args)
    calls, elapsed = 0, 0.0
    while elapsed < budget:
        start = time.perf_counter()
        for _ in range(64):
            fn(*args)
        elapsed += time.perf_counter() - start
        calls += 64
    return elapsed / calls * 1e6


# ----------------------------------------------------------------------
# numpy reference implementations
# ----------------------------------------------------------------------


def generate_actions_np(grid):
    return (grid[0] == 0).astype(np.uint8)


def drop_row_np(grid, action):
    empty = np.flatnonzero(grid[:, action] == 0)
    return int(empty[-1]) if empty.size else -1


def place_token_np(grid, token, action):
    out = grid.copy()
    row = drop_row_np(out, action)
    if row >= 0:
        out[row, action] = token
    return out


def _window_winner(windows):
    if windows.size == 0:
        return 0
    first = windows[..., 0]
    same = (windows == first[..., None]).all(axis=-1) & (first != 0)
    return int(first[same][0]) if same.any() else 0


def winner_np(grid, k):
    """Vectorized full-board scan.

    Loses badly to the compiled version because it must materialize every
    window before it can test any of them, while the compiled scan stops at the
    first line it finds.
    """
    rows, cols = grid.shape
    if cols >= k and (found := _window_winner(sliding_window_view(grid, k, axis=1))):
        return found
    if rows >= k and (found := _window_winner(sliding_window_view(grid, k, axis=0))):
        return found
    if rows >= k and cols >= k:
        boxes = sliding_window_view(grid, (k, k))
        index = np.arange(k)
        if found := _window_winner(boxes[..., index, index]):
            return found
        if found := _window_winner(boxes[..., index, k - 1 - index]):
            return found
    return 0


def winner_at_np(grid, k, row, col):
    """Four lines through one cell. Pure Python beats numpy at this size."""
    rows, cols = grid.shape
    token = grid[row, col]
    if token == 0:
        return 0
    for d_row, d_col in ((0, 1), (1, 0), (1, 1), (1, -1)):
        count = 1
        for sign in (1, -1):
            r, c = row + sign * d_row, col + sign * d_col
            while 0 <= r < rows and 0 <= c < cols and grid[r, c] == token:
                count += 1
                r += sign * d_row
                c += sign * d_col
        if count >= k:
            return int(token)
    return 0


class VecGameNumpy:
    """Batched stepper without numba, for comparison against VecGame."""

    def __init__(self, config, num_envs):
        self.rows, self.cols = config["shape"]
        self.k = config["k"]
        self.players = np.array(config["players"], dtype=np.uint8)
        self.n = num_envs
        self.grids = np.zeros((num_envs, self.rows, self.cols), dtype=np.uint8)
        self.heights = np.zeros((num_envs, self.cols), dtype=np.int64)
        self.actives = np.zeros(num_envs, dtype=np.int64)
        self.times = np.zeros(num_envs, dtype=np.int64)
        self.winners = np.zeros(num_envs, dtype=np.uint8)
        self.filled = np.zeros(num_envs, dtype=np.int64)

    def masks(self):
        return (self.heights < self.rows).astype(np.uint8)

    def _batch_winner(self):
        found = np.zeros(self.n, dtype=np.uint8)
        k = self.k

        def scan(windows):
            first = windows[..., 0]
            same = (windows == first[..., None]).all(axis=-1) & (first != 0)
            flat = same.reshape(self.n, -1)
            hit = flat.any(axis=1)
            if hit.any():
                tokens = first.reshape(self.n, -1)[np.arange(self.n), flat.argmax(axis=1)]
                fill = hit & (found == 0)
                found[fill] = tokens[fill]

        if self.cols >= k:
            scan(sliding_window_view(self.grids, k, axis=2))
        if self.rows >= k:
            scan(sliding_window_view(self.grids, k, axis=1))
        if self.rows >= k and self.cols >= k:
            boxes = sliding_window_view(self.grids, (k, k), axis=(1, 2))
            index = np.arange(k)
            scan(boxes[..., index, index])
            scan(boxes[..., index, k - 1 - index])
        return found

    def step(self, actions):
        env = np.arange(self.n)
        columns = np.clip(actions, 0, self.cols - 1)
        legal = (actions >= 0) & (actions < self.cols)
        legal &= self.heights[env, columns] < self.rows
        rows_idx = self.rows - 1 - self.heights[env, columns]
        tokens = self.players[self.actives]
        self.grids[env[legal], rows_idx[legal], columns[legal]] = tokens[legal]
        self.heights[env[legal], columns[legal]] += 1
        self.filled[legal] += 1
        self.times[legal] += 1
        self.actives[legal] = self.times[legal] % self.players.shape[0]
        # No incremental check is available without a compiled inner loop, so
        # the whole batch is rescanned every step. This is the crux of the cost.
        self.winners = self._batch_winner()
        done = (self.winners != 0) | (self.filled >= self.rows * self.cols)
        if done.any():
            reset = np.flatnonzero(done)
            for array in (self.grids, self.heights):
                array[reset] = 0
            for array in (self.actives, self.times, self.winners, self.filled):
                array[reset] = 0
        return done

    def sample_actions(self, rng):
        weights = self.masks().astype(np.float64)
        totals = weights.sum(axis=1, keepdims=True)
        totals[totals == 0] = 1.0
        cumulative = np.cumsum(weights / totals, axis=1)
        return np.argmax(cumulative > rng.random((self.n, 1)), axis=1).astype(np.int64)


# ----------------------------------------------------------------------


def micro(results: dict) -> None:
    print(f"\n### single-call primitives, microseconds per call ({LABEL} build)")
    header = f"{'shape':<10} {'op':<17} {LABEL:>9} {'numpy':>9} {'ratio':>8}"
    print(header)
    print("-" * len(header))
    rows = []
    for shape in SHAPES:
        grid = cxf.create_grid(shape)
        column = shape[1] // 2
        for _ in range(shape[0] // 2):
            grid = cxf.place_token(grid, np.uint8(1), column)
        landed = cxf.drop_row(grid, column) + 1
        win_length = 4
        cases = [
            (
                "generate_actions",
                lambda g=grid: cxf.generate_actions(g),
                lambda g=grid: generate_actions_np(g),
            ),
            (
                "drop_row",
                lambda g=grid, c=column: cxf.drop_row(g, c),
                lambda g=grid, c=column: drop_row_np(g, c),
            ),
            (
                "place_token",
                lambda g=grid, c=column: cxf.place_token(g, np.uint8(1), c),
                lambda g=grid, c=column: place_token_np(g, np.uint8(1), c),
            ),
            (
                "winner_at",
                lambda g=grid, r=landed, c=column, k=win_length: cxf.winner_at(
                    g, k, r, c
                ),
                lambda g=grid, r=landed, c=column, k=win_length: winner_at_np(g, k, r, c),
            ),
            (
                "winner (full scan)",
                lambda g=grid, k=win_length: cxf.winner(g, k),
                lambda g=grid, k=win_length: winner_np(g, k),
            ),
        ]
        for name, ours, theirs in cases:
            mine, other = bench(ours), bench(theirs)
            print(
                f"{str(shape):<10} {name:<17} {mine:9.3f} {other:9.3f} {other / mine:7.2f}x"
            )
            rows.append(
                {
                    "shape": list(shape),
                    "op": name,
                    LABEL: mine,
                    "numpy": other,
                    "ratio": other / mine,
                }
            )
    results["micro"] = rows


def macro(results: dict) -> None:
    print(f"\n### end-to-end throughput ({LABEL} build)")
    rng = np.random.default_rng(0)
    rows = []

    for shape in [(6, 7), (30, 30)]:
        game = Game(make_config(shape, 4, [1, 2]))
        game.start()
        moves, start = 0, time.perf_counter()
        while time.perf_counter() - start < 0.4:
            legal = cxf.valid_action_columns(game.actions)
            if game.terminal() or legal.size == 0:
                game.start()
                continue
            game.transition(int(legal[moves % legal.size]))
            moves += 1
        rate = moves / (time.perf_counter() - start)
        print(f"  Game.transition {str(shape):<9} {rate:12,.0f} moves/s")
        rows.append(
            {"case": f"Game.transition {shape}", "impl": LABEL, "moves_per_second": rate}
        )

    config = preset("connect4")
    for num_envs in (64, 1024):
        for name, env in (
            (LABEL, VecGame(config, num_envs)),
            ("numpy", VecGameNumpy(config, num_envs)),
        ):
            if hasattr(env, "reset"):
                env.reset()
            env.step(env.sample_actions(rng))
            steps, start = 0, time.perf_counter()
            while time.perf_counter() - start < 0.4:
                env.step(env.sample_actions(rng))
                steps += 1
            rate = steps * num_envs / (time.perf_counter() - start)
            print(f"  VecGame n={num_envs:<5} {name:<7} {rate:12,.0f} moves/s")
            rows.append(
                {"case": f"VecGame n={num_envs}", "impl": name, "moves_per_second": rate}
            )
    results["macro"] = rows


def agents_bench(results: dict) -> None:
    from agents import make_agent

    print(f"\n### agent cost for one opening move ({LABEL} build)")
    config = preset("connect4")
    grid = cxf.create_grid(config["shape"])
    state = {"grid": grid, "info": {"active": 0, "time": 0}}
    actions = cxf.generate_actions(grid)
    rows = []
    for spec in ["greedy", "minimax:depth=4", "mcts:simulations=200"]:
        agent = make_agent(spec, config, seed=0)
        agent.select(state, actions)
        agent.reset(config)
        start = time.perf_counter()
        agent.select(state, actions)
        elapsed = (time.perf_counter() - start) * 1e3
        print(f"  {spec:<24} {elapsed:9.3f} ms")
        rows.append({"agent": spec, "impl": LABEL, "ms_per_move": elapsed})
    results["agents"] = rows


def startup(results: dict) -> None:
    """Import cost is what numba charges on every process start."""
    print("\n### startup cost")
    rows = []
    for name, env in (
        ("with numba", {}),
        ("NUMBA_DISABLE_JIT=1", {"NUMBA_DISABLE_JIT": "1"}),
    ):
        best = min(_time_import(env) for _ in range(3))
        print(f"  warm import, {name:<22} {best:6.2f} s")
        rows.append({"case": name, "seconds": best})
    print("  (a cold import, with numba's on-disk cache cleared, costs ~2 s)")
    results["startup"] = rows


def _time_import(extra_env: dict) -> float:
    import subprocess

    environment = {**os.environ, **extra_env}
    start = time.perf_counter()
    subprocess.run(
        [sys.executable, "-c", "import connectx, agents"],
        env=environment,
        check=True,
        capture_output=True,
    )
    return time.perf_counter() - start


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "sections",
        nargs="*",
        default=None,
        choices=["micro", "macro", "agents", "startup"],
        help="which benchmarks to run (default: all)",
    )
    parser.add_argument("--json", help="write raw results to this path")
    args = parser.parse_args()
    sections = args.sections or ["micro", "macro", "agents", "startup"]

    print(f"=== numba {'ENABLED' if JIT_ENABLED else 'DISABLED'} ===")
    results: dict = {"jit": JIT_ENABLED}
    if "micro" in sections:
        micro(results)
    if "macro" in sections:
        macro(results)
    if "agents" in sections:
        agents_bench(results)
    if "startup" in sections:
        startup(results)

    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(results, indent=2))
        print(f"\nwrote {path}")

    print(
        "\nRun again with NUMBA_DISABLE_JIT=1 to compare against the same code "
        "uncompiled."
        if JIT_ENABLED
        else "\nRun again without NUMBA_DISABLE_JIT to compare against the compiled build."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
