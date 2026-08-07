import numpy as np
import pytest

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

    @pytest.mark.parametrize("shape", [(300, 7), (6, 400), (256, 256)])
    def test_dimensions_above_255_are_not_truncated(self, shape) -> None:
        # Regression: shape entries were typed uint8, so (300, 7) silently
        # became a (44, 7) board.
        assert cxf.create_grid(shape).shape == shape


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
        for _ in range(3):
            grid = cxf.place_token(grid, 1, 0)
        assert grid[shape[0] - 1, 0] == 1
        assert grid[shape[0] - 2, 0] == 1
        assert grid[shape[0] - 3, 0] == 1

    def test_does_not_mutate_input(self, default_config: Config) -> None:
        grid = cxf.create_grid(default_config["shape"])
        original = np.copy(grid)
        cxf.place_token(grid, 1, 0)
        np.testing.assert_array_equal(grid, original)

    def test_full_column_is_no_op(self, default_config: Config) -> None:
        shape = default_config["shape"]
        grid = cxf.create_grid(shape)
        for _ in range(shape[0]):
            grid = cxf.place_token(grid, 1, 0)
        before = np.copy(grid)
        np.testing.assert_array_equal(cxf.place_token(grid, 1, 0), before)

    def test_out_of_range_column_is_no_op(self, default_config: Config) -> None:
        grid = cxf.create_grid(default_config["shape"])
        np.testing.assert_array_equal(cxf.place_token(grid, 1, 99), grid)

    def test_accepts_readonly_grid(self, default_config: Config) -> None:
        # Game.state hands out non-writeable views; every primitive must take one.
        grid = cxf.create_grid(default_config["shape"])
        view = grid.view()
        view.setflags(write=False)
        assert cxf.place_token(view, 1, 0)[default_config["shape"][0] - 1, 0] == 1


class TestDropRowAndLegality:
    def test_drop_row_descends(self, small_config: Config) -> None:
        rows = small_config["shape"][0]
        grid = cxf.create_grid(small_config["shape"])
        for expected in range(rows - 1, -1, -1):
            assert cxf.drop_row(grid, 0) == expected
            grid = cxf.place_token(grid, 1, 0)
        assert cxf.drop_row(grid, 0) == -1

    def test_is_legal_matches_drop_row(self, small_config: Config) -> None:
        grid = cxf.create_grid(small_config["shape"])
        for _ in range(small_config["shape"][0]):
            grid = cxf.place_token(grid, 1, 1)
        assert not cxf.is_legal(grid, 1)
        assert cxf.is_legal(grid, 0)
        assert not cxf.is_legal(grid, -1)
        assert not cxf.is_legal(grid, 99)


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

    def test_full_grid_no_actions(self, small_config: Config) -> None:
        shape = small_config["shape"]
        grid = cxf.create_grid(shape)
        for col in range(shape[1]):
            for _ in range(shape[0]):
                grid = cxf.place_token(grid, 1, col)
        assert np.sum(cxf.generate_actions(grid)) == 0

    def test_valid_action_columns(self) -> None:
        mask = np.array([1, 0, 1, 1, 0], dtype=np.uint8)
        np.testing.assert_array_equal(cxf.valid_action_columns(mask), [0, 2, 3])


class TestWinner:
    def test_empty_has_no_winner(self, default_config: Config) -> None:
        grid = cxf.create_grid(default_config["shape"])
        assert cxf.winner(grid, default_config["k"]) == 0
        assert not cxf.terminal(grid, default_config["k"])

    def test_horizontal(self, default_config: Config) -> None:
        shape, k = default_config["shape"], default_config["k"]
        grid = cxf.create_grid(shape)
        for col in range(k):
            grid = cxf.place_token(grid, 1, col)
        assert cxf.winner(grid, k) == 1

    def test_vertical(self, default_config: Config) -> None:
        shape, k = default_config["shape"], default_config["k"]
        grid = cxf.create_grid(shape)
        for _ in range(k):
            grid = cxf.place_token(grid, 2, 0)
        assert cxf.winner(grid, k) == 2

    def test_diagonal(self, small_config: Config) -> None:
        shape, k = small_config["shape"], small_config["k"]
        grid = cxf.create_grid(shape)
        for i in range(k):
            for _ in range(i):
                grid = cxf.place_token(grid, 2, i)
            grid = cxf.place_token(grid, 1, i)
        assert cxf.winner(grid, k) == 1

    def test_anti_diagonal(self) -> None:
        grid = np.array(
            [
                [0, 0, 1, 0],
                [0, 1, 2, 0],
                [1, 2, 2, 0],
            ],
            dtype=np.uint8,
        )
        assert cxf.winner(grid, 3) == 1

    def test_run_shorter_than_k_is_not_a_win(self, default_config: Config) -> None:
        grid = cxf.create_grid(default_config["shape"])
        for col in range(default_config["k"] - 1):
            grid = cxf.place_token(grid, 1, col)
        assert cxf.winner(grid, default_config["k"]) == 0

    def test_winner_at_matches_full_scan(self, small_config: Config) -> None:
        rng = np.random.default_rng(0)
        shape, k = small_config["shape"], small_config["k"]
        for _ in range(200):
            grid = cxf.create_grid(shape)
            token = 1
            while True:
                legal = cxf.valid_action_columns(cxf.generate_actions(grid))
                if legal.size == 0:
                    break
                column = int(rng.choice(legal))
                row = cxf.drop_row(grid, column)
                grid = cxf.place_token(grid, np.uint8(token), column)
                incremental = cxf.winner_at(grid, k, row, column)
                full_scan = cxf.winner(grid, k)
                assert incremental == full_scan
                if full_scan != 0:
                    break
                token = 2 if token == 1 else 1

    def test_winner_line_traces_the_win(self) -> None:
        grid = np.zeros((4, 5), dtype=np.uint8)
        grid[3, 1:4] = 1
        token, r0, c0, r1, c1 = (int(v) for v in cxf.winner_line(grid, 3))
        assert token == 1
        assert (r0, c0) == (3, 1)
        assert (r1, c1) == (3, 3)

    def test_winner_line_absent(self, default_config: Config) -> None:
        grid = cxf.create_grid(default_config["shape"])
        assert int(cxf.winner_line(grid, default_config["k"])[0]) == -1


class TestTerminalAndDraw:
    def test_full_board_is_terminal(self, small_config: Config) -> None:
        shape = small_config["shape"]
        grid = cxf.create_grid(shape)
        players = [1, 2]
        for c in range(shape[1]):
            for i in range(shape[0]):
                grid = cxf.place_token(grid, players[(c + i) % 2], c)
        assert cxf.full(grid)
        assert cxf.terminal(grid, small_config["k"])

    def test_draw_requires_no_winner(self) -> None:
        # A full board that also contains a line is a win, not a draw.
        grid = np.array([[1, 1, 1, 1]], dtype=np.uint8)
        assert cxf.full(grid)
        assert cxf.winner(grid, 4) == 1
        assert not cxf.is_draw(grid, 4)
        assert cxf.terminal(grid, 4)

    def test_true_draw(self) -> None:
        grid = np.array([[1, 2, 1, 2]], dtype=np.uint8)
        assert cxf.is_draw(grid, 4)
