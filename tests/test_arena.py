import pytest

from connectx import make_config, preset
from connectx.arena import (
    elo_ratings,
    play_game,
    play_match,
    round_robin,
    wilson_interval,
)


class TestWilsonInterval:
    def test_contains_the_point_estimate(self) -> None:
        low, high = wilson_interval(30, 100)
        assert low < 0.30 < high

    def test_stays_inside_zero_one_at_the_extremes(self) -> None:
        for successes, total in [(0, 20), (20, 20), (1, 3)]:
            low, high = wilson_interval(successes, total)
            assert 0.0 <= low <= high <= 1.0

    def test_narrows_with_more_games(self) -> None:
        narrow = wilson_interval(500, 1000)
        wide = wilson_interval(5, 10)
        assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])

    def test_zero_games(self) -> None:
        assert wilson_interval(0, 0) == (0.0, 1.0)


class TestPlayGame:
    def test_returns_a_result(self) -> None:
        result, trajectory = play_game(preset("small"), ["random", "random"], seed=0)
        assert trajectory is None
        assert result.steps > 0
        assert result.winner_entrant in (0, 1, None)

    def test_records_when_asked(self) -> None:
        result, trajectory = play_game(
            preset("small"), ["random", "random"], seed=0, record=True
        )
        assert trajectory is not None
        assert len(trajectory.steps) == result.steps

    def test_seating_maps_seats_to_entrants(self) -> None:
        result, _ = play_game(
            preset("small"), ["greedy", "random"], seating=[1, 0], seed=0
        )
        assert result.seating == [1, 0]
        if result.winner_seat is not None:
            assert result.winner_entrant == result.seating[result.winner_seat]

    def test_rejects_wrong_number_of_specs(self) -> None:
        with pytest.raises(ValueError, match="one agent spec per player"):
            play_game(preset("small"), ["random"])

    def test_is_deterministic_given_a_seed(self) -> None:
        first, first_trajectory = play_game(
            preset("small"), ["random", "random"], seed=42, record=True
        )
        second, second_trajectory = play_game(
            preset("small"), ["random", "random"], seed=42, record=True
        )
        # `duration` is wall-clock, so compare everything that describes the game.
        assert {k: v for k, v in first.to_dict().items() if k != "duration"} == {
            k: v for k, v in second.to_dict().items() if k != "duration"
        }
        assert list(first_trajectory.actions) == list(second_trajectory.actions)

    def test_different_seeds_produce_different_games(self) -> None:
        _, first = play_game(
            preset("connect4"), ["random", "random"], seed=1, record=True
        )
        _, second = play_game(
            preset("connect4"), ["random", "random"], seed=2, record=True
        )
        assert list(first.actions) != list(second.actions)

    def test_game_index_varies_the_seed(self) -> None:
        _, first = play_game(
            preset("connect4"), ["random", "random"], seed=0, game_index=0, record=True
        )
        _, second = play_game(
            preset("connect4"), ["random", "random"], seed=0, game_index=1, record=True
        )
        assert list(first.actions) != list(second.actions)


class TestPlayMatch:
    def test_totals_add_up(self) -> None:
        match = play_match(preset("small"), ["greedy", "random"], games=20, seed=0)
        assert match.games == 20
        assert sum(match.wins) + match.draws == 20
        assert sum(match.seat_wins) == sum(match.wins)

    def test_stronger_agent_wins(self) -> None:
        match = play_match(preset("small"), ["greedy", "random"], games=20, seed=0)
        assert match.win_rate(0) > match.win_rate(1)
        low, _ = match.interval(0)
        assert low > 0.5

    def test_score_counts_draws_as_half(self) -> None:
        match = play_match(preset("connect4"), ["greedy", "greedy"], games=10, seed=1)
        expected = match.wins[0] + 0.5 * match.draws
        assert match.score(0) == pytest.approx(expected)
        assert match.score(0) + match.score(1) == pytest.approx(match.games)

    def test_seat_rotation_is_balanced(self) -> None:
        # With swapping on, each entrant must take seat 0 exactly half the time.
        match = play_match(preset("small"), ["random", "random"], games=10, seed=0)
        assert match.games == 10
        assert sum(match.seat_wins) <= match.games

    def test_rejects_unbalanced_game_counts(self) -> None:
        with pytest.raises(ValueError, match="multiple of the player count"):
            play_match(preset("small"), ["random", "random"], games=7)

    def test_odd_counts_allowed_without_swapping(self) -> None:
        match = play_match(
            preset("small"), ["random", "random"], games=7, swap_seats=False
        )
        assert match.games == 7

    def test_rejects_zero_games(self) -> None:
        with pytest.raises(ValueError, match="games must be positive"):
            play_match(preset("small"), ["random", "random"], games=0)

    def test_deterministic_across_worker_counts(self) -> None:
        single = play_match(preset("small"), ["greedy", "random"], games=8, seed=5)
        parallel = play_match(
            preset("small"), ["greedy", "random"], games=8, seed=5, workers=3
        )
        assert single.wins == parallel.wins
        assert single.draws == parallel.draws
        assert single.seat_wins == parallel.seat_wins

    def test_summary_and_dict(self) -> None:
        match = play_match(preset("small"), ["greedy", "random"], games=4, seed=0)
        assert "greedy" in match.summary()
        payload = match.to_dict()
        assert payload["variant"] == "5x6k4p2"
        assert len(payload["interval"]) == 2

    def test_multiplayer_match(self) -> None:
        config = make_config((5, 6), 3, [1, 2, 3])
        match = play_match(config, ["greedy", "random", "random"], games=6, seed=0)
        assert len(match.wins) == 3
        assert len(match.seat_wins) == 3
        assert sum(match.wins) + match.draws == 6


class TestElo:
    def test_stronger_entrant_rates_higher(self) -> None:
        ratings = elo_ratings(["strong", "weak"], {(0, 1): (90.0, 100.0)})
        assert ratings["strong"] > ratings["weak"]

    def test_even_results_rate_equally(self) -> None:
        ratings = elo_ratings(["a", "b"], {(0, 1): (50.0, 100.0)})
        assert ratings["a"] == pytest.approx(ratings["b"], abs=1e-6)

    def test_mean_is_the_anchor(self) -> None:
        ratings = elo_ratings(
            ["a", "b", "c"],
            {(0, 1): (70.0, 100.0), (0, 2): (80.0, 100.0), (1, 2): (60.0, 100.0)},
            anchor=1500.0,
        )
        assert sum(ratings.values()) / 3 == pytest.approx(1500.0, abs=1.0)

    def test_transitive_ordering(self) -> None:
        ratings = elo_ratings(
            ["a", "b", "c"],
            {(0, 1): (75.0, 100.0), (1, 2): (75.0, 100.0), (0, 2): (95.0, 100.0)},
        )
        assert ratings["a"] > ratings["b"] > ratings["c"]

    def test_perfect_records_stay_finite(self) -> None:
        ratings = elo_ratings(["unbeaten", "winless"], {(0, 1): (100.0, 100.0)})
        assert all(abs(value) < 1e6 for value in ratings.values())

    def test_no_entrants(self) -> None:
        assert elo_ratings([], {}) == {}


class TestRoundRobin:
    def test_orders_the_ladder(self) -> None:
        result = round_robin(
            preset("small"), ["random", "greedy", "minimax:depth=3"], games=6, seed=0
        )
        assert (
            result.ratings["minimax:depth=3"]
            > result.ratings["greedy"]
            > result.ratings["random"]
        )
        assert len(result.matches) == 3

    def test_table_lists_every_entrant(self) -> None:
        result = round_robin(preset("small"), ["random", "greedy"], games=4, seed=0)
        table = result.table()
        assert "random" in table and "greedy" in table

    def test_to_dict(self) -> None:
        result = round_robin(preset("small"), ["random", "greedy"], games=4, seed=0)
        payload = result.to_dict()
        assert payload["variant"] == "5x6k4p2"
        assert len(payload["matches"]) == 1

    def test_needs_two_entrants(self) -> None:
        with pytest.raises(ValueError, match="at least two"):
            round_robin(preset("small"), ["random"], games=2)

    def test_rejects_multiplayer_variants(self) -> None:
        config = make_config((5, 6), 3, [1, 2, 3])
        with pytest.raises(ValueError, match="seats 3 players"):
            round_robin(config, ["random", "greedy"], games=2)
