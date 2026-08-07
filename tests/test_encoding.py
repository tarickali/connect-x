import numpy as np

from connectx import Game, make_config, preset
from connectx.encoding import (
    action_mask,
    encode_batch,
    encode_grid,
    encode_state,
    mirror_action,
    mirror_grid,
    mirror_observation,
    observation_shape,
)


class TestObservationShape:
    def test_two_players(self) -> None:
        assert observation_shape(preset("connect4")) == (3, 6, 7)

    def test_scales_with_players_not_board(self) -> None:
        assert observation_shape(make_config((9, 9), 4, [1, 2, 3])) == (4, 9, 9)


class TestEncodeGrid:
    def test_planes_are_one_hot_over_tokens(self) -> None:
        grid = np.array([[0, 1, 2]], dtype=np.uint8)
        planes = encode_grid(grid, [1, 2], perspective=0)
        np.testing.assert_array_equal(planes[0], [[0, 1, 0]])
        np.testing.assert_array_equal(planes[1], [[0, 0, 1]])
        np.testing.assert_array_equal(planes[2], [[1, 0, 0]])

    def test_perspective_rotates_planes(self) -> None:
        grid = np.array([[0, 1, 2]], dtype=np.uint8)
        first = encode_grid(grid, [1, 2], perspective=0)
        second = encode_grid(grid, [1, 2], perspective=1)
        np.testing.assert_array_equal(first[0], second[1])
        np.testing.assert_array_equal(first[1], second[0])
        np.testing.assert_array_equal(first[2], second[2])

    def test_planes_partition_the_board(self) -> None:
        grid = np.array([[0, 1, 2, 3]], dtype=np.uint8)
        planes = encode_grid(grid, [1, 2, 3])
        np.testing.assert_allclose(planes.sum(axis=0), np.ones((1, 4)))

    def test_dtype_is_float32(self) -> None:
        planes = encode_grid(np.zeros((2, 2), np.uint8), [1, 2])
        assert planes.dtype == np.float32


class TestEncodeState:
    def test_defaults_to_the_player_to_move(self) -> None:
        config = preset("connect4")
        game = Game(config)
        game.start()
        game.transition(3)
        # Seat 1 is to move, so its own tokens must land in plane 0.
        planes = encode_state(game.state, config)
        assert planes[0].sum() == 0
        assert planes[1].sum() == 1

    def test_accepts_readonly_grids(self) -> None:
        config = preset("connect4")
        game = Game(config)
        game.start()
        assert encode_state(game.state, config).shape == observation_shape(config)


class TestActionMask:
    def test_converts_to_bool(self) -> None:
        mask = action_mask(np.array([1, 0, 1], dtype=np.uint8))
        assert mask.dtype == bool
        np.testing.assert_array_equal(mask, [True, False, True])


class TestMirror:
    def test_grid_reflects(self) -> None:
        grid = np.array([[1, 0, 2]], dtype=np.uint8)
        np.testing.assert_array_equal(mirror_grid(grid), [[2, 0, 1]])

    def test_mirror_is_an_involution(self) -> None:
        grid = np.array([[1, 0, 2, 2]], dtype=np.uint8)
        np.testing.assert_array_equal(mirror_grid(mirror_grid(grid)), grid)

    def test_action_maps_consistently(self) -> None:
        for column in range(7):
            assert mirror_action(mirror_action(column, 7), 7) == column
        assert mirror_action(0, 7) == 6
        assert mirror_action(3, 7) == 3

    def test_mirroring_a_move_matches_mirroring_the_result(self) -> None:
        # Playing column c then mirroring == mirroring then playing mirror(c).
        import connectx.functional as cxf

        grid = cxf.create_grid((4, 5))
        grid = cxf.place_token(grid, np.uint8(1), 1)
        played_then_mirrored = mirror_grid(cxf.place_token(grid, np.uint8(2), 3))
        mirrored_then_played = cxf.place_token(
            mirror_grid(grid), np.uint8(2), mirror_action(3, 5)
        )
        np.testing.assert_array_equal(played_then_mirrored, mirrored_then_played)

    def test_observation_mirror_preserves_planes(self) -> None:
        planes = encode_grid(np.array([[1, 0, 2]], dtype=np.uint8), [1, 2])
        mirrored = mirror_observation(planes)
        assert mirrored.shape == planes.shape
        np.testing.assert_array_equal(mirrored[0], [[0, 0, 1]])


class TestEncodeBatch:
    def test_passthrough_without_mirror(self) -> None:
        batch = np.zeros((4, 3, 2, 2), dtype=np.float32)
        assert encode_batch(batch).shape == (4, 3, 2, 2)

    def test_mirror_doubles_the_batch(self) -> None:
        batch = np.zeros((4, 3, 2, 2), dtype=np.float32)
        assert encode_batch(batch, mirror=True).shape == (8, 3, 2, 2)

    def test_empty_batch(self) -> None:
        batch = np.zeros((0, 3, 2, 2), dtype=np.float32)
        assert encode_batch(batch, mirror=True).shape == (0, 3, 2, 2)
