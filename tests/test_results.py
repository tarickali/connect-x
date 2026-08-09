import json
import tempfile
from pathlib import Path

import pytest

from connectx import make_config, preset
from connectx.arena import (
    elo_to_score,
    games_needed,
    play_match,
    resolvable_gap,
    round_robin,
    score_to_elo,
)
from connectx.results import (
    append_jsonl,
    completed_variants,
    git_revision,
    provenance,
    read_jsonl,
)
from connectx.variants import sweep, tournament_surface, variant_grid


class TestProvenance:
    def test_carries_the_seed(self) -> None:
        assert provenance(seed=17)["seed"] == 17

    def test_records_the_stack(self) -> None:
        record = provenance()
        assert set(record["env"]) >= {"connectx", "python", "numpy", "numba"}
        assert record["timestamp"].endswith("+00:00")

    def test_git_revision_is_safe_outside_a_checkout(self) -> None:
        revision = git_revision()
        assert set(revision) == {"sha", "dirty"}
        assert revision["sha"] is None or len(revision["sha"]) == 40

    def test_extra_fields_pass_through(self) -> None:
        assert provenance(seed=1, note="pilot")["note"] == "pilot"

    def test_is_json_serializable(self) -> None:
        json.dumps(provenance(seed=3))


class TestJsonl:
    def test_append_and_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "out.jsonl"
            append_jsonl(path, {"variant": "a", "value": 1})
            append_jsonl(path, {"variant": "b", "value": 2})
            records = read_jsonl(path)
        assert [r["variant"] for r in records] == ["a", "b"]

    def test_missing_file_reads_empty(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            assert read_jsonl(Path(directory) / "absent.jsonl") == []

    def test_blank_lines_are_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "out.jsonl"
            path.write_text('{"variant": "a"}\n\n\n{"variant": "b"}\n')
            assert len(read_jsonl(path)) == 2

    def test_completed_variants(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "out.jsonl"
            append_jsonl(path, {"variant": "4x5k3p2"})
            append_jsonl(path, {"no_variant_key": True})
            assert completed_variants(path) == {"4x5k3p2"}


class TestResultProvenance:
    def test_match_record_can_be_reproduced_from(self) -> None:
        match = play_match(preset("tiny"), ["greedy", "random"], games=4, seed=11)
        record = match.to_dict()
        assert record["provenance"]["seed"] == 11
        assert record["provenance"]["env"]["connectx"]

    def test_tournament_record_carries_provenance(self) -> None:
        result = round_robin(preset("tiny"), ["random", "greedy"], games=4, seed=5)
        assert result.to_dict()["provenance"]["seed"] == 5

    def test_surface_record_carries_provenance(self) -> None:
        configs = variant_grid([(4, 5)], [3], [2])
        surface = tournament_surface(["random", "greedy"], configs, games=4, seed=9)
        assert surface.to_records()[0]["provenance"]["seed"] == 9

    def test_match_roundtrips_through_its_record(self) -> None:
        match = play_match(preset("tiny"), ["greedy", "random"], games=6, seed=2)
        restored = type(match).from_dict(match.to_dict())
        assert restored.wins == match.wins
        assert restored.draws == match.draws
        assert restored.seat_wins == match.seat_wins
        assert restored.total_steps == match.total_steps
        assert restored.config == match.config
        assert restored.score_rate(0) == match.score_rate(0)


class TestSampleSize:
    def test_elo_score_roundtrip(self) -> None:
        for gap in (-300.0, -25.0, 0.0, 25.0, 300.0):
            assert score_to_elo(elo_to_score(gap)) == pytest.approx(gap, abs=1e-6)

    def test_even_match_is_half(self) -> None:
        assert elo_to_score(0.0) == pytest.approx(0.5)

    def test_bigger_gaps_need_fewer_games(self) -> None:
        counts = [games_needed(gap) for gap in (25, 50, 100, 200, 400)]
        assert counts == sorted(counts, reverse=True)

    def test_counts_are_even_so_seats_stay_balanced(self) -> None:
        for gap in (30, 60, 130, 250):
            assert games_needed(gap) % 2 == 0

    def test_zero_gap_is_unanswerable(self) -> None:
        assert games_needed(0) == 0

    def test_sign_does_not_matter(self) -> None:
        assert games_needed(-120) == games_needed(120)

    def test_higher_confidence_costs_more_games(self) -> None:
        assert games_needed(100, confidence=0.99) > games_needed(100, confidence=0.90)

    def test_resolvable_gap_inverts_games_needed(self) -> None:
        for games in (40, 100, 400, 2000):
            gap = resolvable_gap(games)
            assert games_needed(gap) <= games
            # And a slightly smaller gap should not fit in the budget.
            assert games_needed(gap * 0.9) > games

    def test_tiny_budgets_resolve_nothing(self) -> None:
        assert resolvable_gap(1) == float("inf")

    def test_documented_numbers_hold(self) -> None:
        # These appear in the docs; keep them honest.
        assert 90 <= games_needed(100) <= 110
        assert 150 <= resolvable_gap(40) <= 170


class TestResume:
    def test_sweep_writes_incrementally(self) -> None:
        configs = variant_grid([(4, 5), (5, 6)], [3], [2])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sweep.jsonl"
            seen: list[int] = []
            sweep(
                ["greedy", "random"],
                configs,
                games=4,
                seed=0,
                jsonl=path,
                on_variant=lambda _m: seen.append(len(read_jsonl(path))),
            )
            # Each record lands before the next variant starts.
            assert seen == [1, 2]

    def test_sweep_resumes_and_skips_finished_variants(self) -> None:
        configs = variant_grid([(4, 5), (5, 6)], [3], [2])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sweep.jsonl"
            first = sweep(["greedy", "random"], configs[:1], games=4, seed=0, jsonl=path)
            replayed: list[str] = []
            second = sweep(
                ["greedy", "random"],
                configs,
                games=4,
                seed=0,
                jsonl=path,
                resume=True,
                on_variant=lambda m: replayed.append(m.config["shape"][0]),
            )
        # Only the unfinished variant was played again.
        assert replayed == [5]
        assert second.variants == ["4x5k3p2", "5x6k3p2"]
        # The resumed entry came back off disk intact.
        assert second.matches[0].wins == first.matches[0].wins

    def test_resume_on_a_complete_file_plays_nothing(self) -> None:
        configs = variant_grid([(4, 5)], [3], [2])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sweep.jsonl"
            sweep(["greedy", "random"], configs, games=4, seed=0, jsonl=path)
            played: list[str] = []
            result = sweep(
                ["greedy", "random"],
                configs,
                games=4,
                seed=0,
                jsonl=path,
                resume=True,
                on_variant=lambda m: played.append("ran"),
            )
        assert played == []
        assert len(result.matches) == 1

    def test_resume_is_opt_in(self) -> None:
        configs = variant_grid([(4, 5)], [3], [2])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sweep.jsonl"
            sweep(["greedy", "random"], configs, games=4, seed=0, jsonl=path)
            sweep(["greedy", "random"], configs, games=4, seed=0, jsonl=path)
            # Without resume the variant is replayed and appended again.
            assert len(read_jsonl(path)) == 2

    def test_surface_resumes(self) -> None:
        configs = variant_grid([(4, 5), (5, 6)], [3], [2])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "surface.jsonl"
            first = tournament_surface(
                ["random", "greedy"], configs[:1], games=4, seed=0, jsonl=path
            )
            played: list[str] = []
            second = tournament_surface(
                ["random", "greedy"],
                configs,
                games=4,
                seed=0,
                jsonl=path,
                resume=True,
                on_variant=lambda _t: played.append("ran"),
            )
        assert len(played) == 1
        assert second.variants == ["4x5k3p2", "5x6k3p2"]
        assert second.rating_matrix()["greedy"]["4x5k3p2"] == pytest.approx(
            first.rating_matrix()["greedy"]["4x5k3p2"]
        )


class TestEngineFactoryPlumbing:
    def test_play_match_accepts_an_engine(self) -> None:
        from connectx import Game

        match = play_match(
            preset("tiny"), ["random", "random"], games=4, seed=0, engine=Game
        )
        assert match.games == 4

    def test_sweep_accepts_an_engine(self) -> None:
        from connectx import Game

        configs = variant_grid([(4, 5)], [3], [2])
        result = sweep(["greedy", "random"], configs, games=4, seed=0, engine=Game)
        assert len(result.matches) == 1

    def test_adapters_accept_an_engine(self) -> None:
        pytest.importorskip("gymnasium")
        pytest.importorskip("pettingzoo")
        from connectx import Game
        from connectx.adapters import make_aec_env, make_gym_env

        make_aec_env(preset("tiny"), engine=Game).reset(seed=0)
        make_gym_env(preset("tiny"), engine=Game).reset(seed=0)

    def test_action_space_size_drives_policy_width(self) -> None:
        from connectx import Game
        from connectx.dataset import build_supervised

        config = make_config((5, 9), 4, [1, 2])
        game = Game(config, record=True)
        game.start()
        for column in (0, 1, 2, 3):
            game.transition(column)
        data = build_supervised([game.trajectory()])
        assert data["policies"].shape[1] == Game(config).action_space_size == 9
