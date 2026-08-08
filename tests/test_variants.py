import json
import tempfile
from pathlib import Path

import pytest

from connectx.config import config_id
from connectx.variants import sweep, variant_grid


class TestVariantGrid:
    def test_crosses_every_dimension(self) -> None:
        configs = variant_grid([(6, 7), (8, 9)], [3, 4], [2])
        assert len(configs) == 4
        assert {config_id(c) for c in configs} == {
            "6x7k3p2",
            "6x7k4p2",
            "8x9k3p2",
            "8x9k4p2",
        }

    def test_drops_unwinnable_combinations(self) -> None:
        # k=9 does not fit on a 4x5 board, so that corner disappears silently.
        configs = variant_grid([(4, 5), (10, 10)], [9], [2])
        assert [config_id(c) for c in configs] == ["10x10k9p2"]

    def test_player_counts_produce_token_lists(self) -> None:
        configs = variant_grid([(6, 7)], [4], [2, 3, 4])
        assert [c["players"] for c in configs] == [[1, 2], [1, 2, 3], [1, 2, 3, 4]]

    def test_empty_grid(self) -> None:
        assert variant_grid([(3, 3)], [10], [2]) == []


class TestSweep:
    def test_runs_every_variant(self) -> None:
        configs = variant_grid([(4, 5), (5, 6)], [3], [2])
        result = sweep(["greedy", "random"], configs, games=4, seed=0)
        assert len(result.matches) == 2
        assert result.variants == ["4x5k3p2", "5x6k3p2"]
        assert not result.skipped

    def test_skips_seat_count_mismatches(self) -> None:
        configs = variant_grid([(5, 6)], [3], [2, 3])
        result = sweep(["greedy", "random"], configs, games=4, seed=0)
        assert result.variants == ["5x6k3p2"]
        assert result.skipped == ["5x6k3p3"]

    def test_score_rates_cover_every_variant(self) -> None:
        configs = variant_grid([(4, 5), (5, 6)], [3], [2])
        result = sweep(["greedy", "random"], configs, games=4, seed=0)
        rates = result.score_rates(0)
        assert set(rates) == {"4x5k3p2", "5x6k3p2"}
        assert all(0.0 <= value <= 1.0 for value in rates.values())

    def test_reordering_variants_does_not_change_a_result(self) -> None:
        first = sweep(
            ["greedy", "random"], variant_grid([(5, 6)], [3], [2]), games=4, seed=0
        )
        second = sweep(
            ["greedy", "random"],
            variant_grid([(5, 6), (6, 7)], [3], [2]),
            games=4,
            seed=0,
        )
        assert first.matches[0].wins == second.matches[0].wins

    def test_callback_fires_per_variant(self) -> None:
        seen = []
        configs = variant_grid([(4, 5), (5, 6)], [3], [2])
        sweep(
            ["greedy", "random"],
            configs,
            games=4,
            seed=0,
            on_variant=lambda match: seen.append(config_id(match.config)),
        )
        assert seen == ["4x5k3p2", "5x6k3p2"]

    def test_table_renders(self) -> None:
        configs = variant_grid([(4, 5)], [3], [2])
        result = sweep(["greedy", "random"], configs, games=4, seed=0)
        table = result.table()
        assert "4x5k3p2" in table
        assert "greedy" in table

    def test_empty_table(self) -> None:
        assert sweep(["greedy", "random"], [], games=4).table() == "(no variants)"

    def test_write_jsonl(self) -> None:
        configs = variant_grid([(4, 5), (5, 6)], [3], [2])
        result = sweep(["greedy", "random"], configs, games=4, seed=0)
        with tempfile.TemporaryDirectory() as directory:
            path = result.write_jsonl(Path(directory) / "nested" / "out.jsonl")
            lines = path.read_text().strip().splitlines()
        assert len(lines) == 2
        record = json.loads(lines[0])
        assert record["variant"] == "4x5k3p2"
        assert record["specs"] == ["greedy", "random"]


class TestTournamentSurface:
    def _surface(self, games: int = 4):
        from connectx.variants import tournament_surface

        configs = variant_grid([(4, 5), (5, 6)], [3], [2])
        return tournament_surface(
            ["random", "greedy", "minimax:depth=2"], configs, games=games, seed=0
        )

    def test_runs_a_round_robin_per_variant(self) -> None:
        surface = self._surface()
        assert surface.variants == ["4x5k3p2", "5x6k3p2"]
        assert len(surface.tournaments) == 2
        assert not surface.skipped

    def test_skips_multiplayer_variants(self) -> None:
        from connectx.variants import tournament_surface

        configs = variant_grid([(5, 6)], [3], [2, 3])
        surface = tournament_surface(["random", "greedy"], configs, games=4, seed=0)
        assert surface.variants == ["5x6k3p2"]
        assert surface.skipped == ["5x6k3p3"]

    def test_rating_matrix_is_dense(self) -> None:
        surface = self._surface()
        matrix = surface.rating_matrix()
        assert set(matrix) == {"random", "greedy", "minimax:depth=2"}
        for ratings in matrix.values():
            assert set(ratings) == {"4x5k3p2", "5x6k3p2"}

    def test_each_column_is_anchored_independently(self) -> None:
        # Ratings are only comparable within a variant, which is exactly what
        # the anchoring guarantees; assert it so the caveat stays true.
        surface = self._surface()
        for tournament in surface.tournaments:
            values = list(tournament.ratings.values())
            assert sum(values) / len(values) == pytest.approx(1500.0, abs=1.0)

    def test_ranks_are_a_permutation(self) -> None:
        surface = self._surface()
        ranks = surface.ranks()
        for variant in surface.variants:
            positions = sorted(ranks[spec][variant] for spec in surface.specs)
            assert positions == [1, 2, 3]

    def test_spread_is_non_negative(self) -> None:
        surface = self._surface()
        assert all(value >= 0 for value in surface.spread().values())

    def test_ladder_order_holds_on_easy_variants(self) -> None:
        surface = self._surface(games=10)
        ranks = surface.ranks()
        for variant in surface.variants:
            assert ranks["minimax:depth=2"][variant] < ranks["random"][variant]

    def test_rank_changes_lists_unstable_agents(self) -> None:
        surface = self._surface()
        moved = surface.rank_changes()
        ranks = surface.ranks()
        for spec in surface.specs:
            unstable = len(set(ranks[spec].values())) > 1
            assert (spec in moved) == unstable

    def test_table_renders(self) -> None:
        table = self._surface().table()
        assert "4x5k3p2" in table
        assert "spread" in table
        assert "ladder order" in table

    def test_empty_surface(self) -> None:
        from connectx.variants import tournament_surface

        assert tournament_surface(["random", "greedy"], []).table() == "(no variants)"

    def test_write_jsonl(self) -> None:
        surface = self._surface()
        with tempfile.TemporaryDirectory() as directory:
            path = surface.write_jsonl(Path(directory) / "surface.jsonl")
            lines = path.read_text().strip().splitlines()
        assert len(lines) == 2
        record = json.loads(lines[0])
        assert record["variant"] == "4x5k3p2"
        assert set(record["ratings"]) == {"random", "greedy", "minimax:depth=2"}
        assert set(record["ranks"].values()) == {1, 2, 3}
