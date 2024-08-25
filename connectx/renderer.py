import numpy as np

__all__ = ["terminal_render"]


def terminal_render(grid: np.ndarray, time: int, player: int) -> None:
    print("===" * grid.shape[1])
    print(f"Time: {time} - Player: {player}")
    print(" -" + "--" * grid.shape[1])
    for row in range(grid.shape[0]):
        print("|", end=" ")
        for col in range(grid.shape[1]):
            print(grid[row][col], end=" ")
        print("|")
    print(" -" + "--" * grid.shape[1])
    print("===" * grid.shape[1])
