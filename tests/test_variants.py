import json
import tempfile
from pathlib import Path

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
