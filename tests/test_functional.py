import numpy as np

import connectx.functional as cxf
from connectx.types import Config


class TestCreateGrid:
    def test_shape(self, default_config: Config) -> None:
        shape = default_config["shape"]
        grid = cxf.create_grid(shape)
        assert grid.shape == shape
        assert grid.dtype == np.uint8

    def test_empty(self, default_config: Config) -> None:
        grid = cxf.create_grid(default_config["shape"])
        assert np.all(grid == 0)


class TestPlaceToken:
    def test_place_bottom_column(self, default_config: Config) -> None:
        shape = default_config["shape"]
        grid = cxf.create_grid(shape)
        out = cxf.place_token(grid, 1, 0)
        assert out[shape[0] - 1, 0] == 1
        assert np.sum(out) == 1

    def test_place_stacks(self, default_config: Config) -> None:
        shape = default_config["shape"]
        grid = cxf.create_grid(shape)
        col = 0
        for _ in range(3):
            grid = cxf.place_token(grid, 1, col)
        assert grid[shape[0] - 1, col] == 1
        assert grid[shape[0] - 2, col] == 1
        assert grid[shape[0] - 3, col] == 1

    def test_does_not_mutate_input(self, default_config: Config) -> None:
        shape = default_config["shape"]
        grid = cxf.create_grid(shape)
        original = np.copy(grid)
        cxf.place_token(grid, 1, 0)
        np.testing.assert_array_equal(grid, original)

    def test_full_column_is_no_op(self, default_config: Config) -> None:
        shape = default_config["shape"]
        grid = cxf.create_grid(shape)
        for _ in range(shape[0]):
            grid = cxf.place_token(grid, 1, 0)
        before = np.copy(grid)
        after = cxf.place_token(grid, 1, 0)
        np.testing.assert_array_equal(after, before)


class TestGenerateActions:
    def test_empty_grid_all_columns_valid(self, default_config: Config) -> None:
        grid = cxf.create_grid(default_config["shape"])
        actions = cxf.generate_actions(grid)
        assert actions.shape == (grid.shape[1],)
        assert np.sum(actions) == grid.shape[1]

    def test_full_column_invalid(self, default_config: Config) -> None:
        shape = default_config["shape"]
        grid = cxf.create_grid(shape)
        for _ in range(shape[0]):
            grid = cxf.place_token(grid, 1, 0)
        actions = cxf.generate_actions(grid)
        assert actions[0] == 0
        assert np.sum(actions) == shape[1] - 1

    def test_full_grid_no_actions(self, default_config: Config) -> None:
        shape = default_config["shape"]
        grid = cxf.create_grid(shape)
        for col in range(shape[1]):
            for _ in range(shape[0]):
                grid = cxf.place_token(grid, 1, col)
        actions = cxf.generate_actions(grid)
        assert np.sum(actions) == 0


class TestTerminal:
    def test_empty_not_terminal(self, default_config: Config) -> None:
        grid = cxf.create_grid(default_config["shape"])
        assert not cxf.terminal(grid, default_config["k"])

    def test_horizontal_win(self, default_config: Config) -> None:
        shape, k = default_config["shape"], default_config["k"]
        grid = cxf.create_grid(shape)
        row = shape[0] - 1
        for col in range(k):
            grid = cxf.place_token(grid, 1, col)
        assert cxf.terminal(grid, k)

    def test_vertical_win(self, default_config: Config) -> None:
        shape, k = default_config["shape"], default_config["k"]
        grid = cxf.create_grid(shape)
        for _ in range(k):
            grid = cxf.place_token(grid, 1, 0)
        assert cxf.terminal(grid, k)

    def test_diagonal_win(self, small_config: Config) -> None:
        shape, k = small_config["shape"], small_config["k"]
        grid = cxf.create_grid(shape)
        # Build a main diagonal of length k (e.g. (0,0), (1,1), (2,2))
        for i in range(k):
            for _ in range(i):
                grid = cxf.place_token(grid, 2, i)  # fill column below
            grid = cxf.place_token(grid, 1, i)
        assert cxf.terminal(grid, k)

    def test_tie(self, small_config: Config) -> None:
        shape = small_config["shape"]
        grid = cxf.create_grid(shape)
        players = [1, 2]
        for c in range(shape[1]):
            for i in range(shape[0]):
                grid = cxf.place_token(grid, players[(c + i) % 2], c)
        assert np.all(grid != 0)
        assert cxf.terminal(grid, small_config["k"])
