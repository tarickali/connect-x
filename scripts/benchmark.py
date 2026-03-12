import time

import connectx.functional as cxf
from connectx import Game
from connectx.utils import make_config


def time_func(f, *args, n: int = 50_000) -> float:
    """Run function f(*args) n times and return mean time per call in microseconds."""
    start = time.perf_counter()
    for _ in range(n):
        f(*args)
    return (time.perf_counter() - start) / n * 1e6


def benchmark_functional() -> None:
    """Benchmark generate_actions and place_token for various board dimensions."""
    shapes = [(6, 7), (8, 9), (10, 12), (12, 14), (15, 18), (100, 100)]
    k = 4
    n_calls = 30_000

    print("Functional API (numba JIT)")
    print("-" * 60)
    print(
        f"{'shape':<12} {'generate_actions (µs/call)':<28} {'place_token (µs/call)':<22}"
    )
    print("-" * 60)

    for shape in shapes:
        grid = cxf.create_grid(shape)
        # Warmup
        cxf.generate_actions(grid)
        cxf.place_token(grid, 1, shape[1] // 2)

        t_actions = time_func(cxf.generate_actions, grid, n=n_calls)
        t_place = time_func(cxf.place_token, grid, 1, shape[1] // 2, n=n_calls)
        print(f"{str(shape):<12} {t_actions:<28.3f} {t_place:<22.3f}")

    print()


def benchmark_game_transition() -> None:
    """Benchmark Game.transition for various board dimensions."""
    shapes = [(6, 7), (8, 9), (10, 12), (12, 14), (15, 18), (100, 100)]
    k = 4
    n_calls = 20_000

    print("Game.transition (place_token + generate_actions) per dimension")
    print("-" * 60)
    print(f"{'shape':<12} {'transition (µs/call)':<22}")
    print("-" * 60)

    for shape in shapes:
        config = make_config(shape, k, [1, 2])
        game = Game(config)
        game.start()
        valid_cols = [c for c in range(shape[1]) if game.actions[c] == 1]

        # Warmup
        game.start()
        game.transition(valid_cols[0])

        game.start()
        start = time.perf_counter()
        for i in range(n_calls):
            game.transition(valid_cols[i % len(valid_cols)])
            if game.terminal():
                game.start()
        elapsed = (time.perf_counter() - start) / n_calls * 1e6
        print(f"{str(shape):<12} {elapsed:<22.3f}")

    print()


if __name__ == "__main__":
    print("connect-x benchmarks")
    benchmark_functional()
    benchmark_game_transition()
