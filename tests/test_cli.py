import json
import tempfile
from pathlib import Path

import pytest

from connectx.cli import _parse_shape, build_parser, main


class TestShapeParsing:
    @pytest.mark.parametrize(
        "text, expected", [("6x7", (6, 7)), ("6,7", (6, 7)), ("12X14", (12, 14))]
    )
    def test_accepts_both_separators(self, text, expected) -> None:
        assert _parse_shape(text) == expected

    @pytest.mark.parametrize("text", ["6", "axb", "6x7x8", ""])
    def test_rejects_junk(self, text) -> None:
        import argparse

        with pytest.raises(argparse.ArgumentTypeError):
            _parse_shape(text)


class TestParser:
    def test_requires_a_subcommand(self) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args([])

    def test_variant_flags_override_the_preset(self) -> None:
        args = build_parser().parse_args(
            [
                "match",
                "--preset",
                "connect4",
                "--shape",
                "5",
                "6",
                "--agents",
                "random",
                "random",
            ]
        )
        assert args.shape == [5, 6]


class TestCommands:
    def test_agents_listing(self, capsys) -> None:
        assert main(["agents"]) == 0
        output = capsys.readouterr().out
        assert "minimax" in output
        assert "connect4" in output

    def test_play(self, capsys) -> None:
        assert (
            main(["play", "--preset", "tiny", "--agents", "random", "random", "--quiet"])
            == 0
        )
        assert "Moves:" in capsys.readouterr().out

    def test_play_rejects_a_seat_mismatch(self, capsys) -> None:
        assert main(["play", "--preset", "tiny", "--agents", "random"]) == 2
        assert "seats" in capsys.readouterr().err

    def test_play_reports_a_winner_or_draw(self, capsys) -> None:
        main(
            [
                "play",
                "--preset",
                "tiny",
                "--agents",
                "greedy",
                "random",
                "--quiet",
                "--seed",
                "0",
            ]
        )
        output = capsys.readouterr().out
        assert "Winner:" in output or "Draw." in output

    def test_match(self, capsys) -> None:
        assert (
            main(
                [
                    "match",
                    "--preset",
                    "tiny",
                    "--agents",
                    "greedy",
                    "random",
                    "--games",
                    "4",
                ]
            )
            == 0
        )
        assert "95% CI" in capsys.readouterr().out

    def test_match_writes_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "out" / "match.json"
            main(
                [
                    "match",
                    "--preset",
                    "tiny",
                    "--agents",
                    "greedy",
                    "random",
                    "--games",
                    "4",
                    "--json",
                    str(path),
                ]
            )
            payload = json.loads(path.read_text())
        assert payload["specs"] == ["greedy", "random"]
        assert payload["games"] == 4

    def test_tournament(self, capsys) -> None:
        assert (
            main(
                [
                    "tournament",
                    "--preset",
                    "tiny",
                    "--agents",
                    "random",
                    "greedy",
                    "--games",
                    "4",
                ]
            )
            == 0
        )
        assert "elo" in capsys.readouterr().out

    def test_sweep(self, capsys) -> None:
        code = main(
            [
                "sweep",
                "--agents",
                "greedy",
                "random",
                "--shapes",
                "4x5",
                "5x6",
                "--ks",
                "3",
                "--games",
                "4",
                "--quiet",
            ]
        )
        assert code == 0
        output = capsys.readouterr().out
        assert "4x5k3p2" in output and "5x6k3p2" in output

    def test_sweep_writes_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sweep.jsonl"
            main(
                [
                    "sweep",
                    "--agents",
                    "greedy",
                    "random",
                    "--shapes",
                    "4x5",
                    "--ks",
                    "3",
                    "--games",
                    "4",
                    "--quiet",
                    "--jsonl",
                    str(path),
                ]
            )
            lines = path.read_text().strip().splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["variant"] == "4x5k3p2"

    def test_sweep_with_no_valid_variants(self, capsys) -> None:
        code = main(
            [
                "sweep",
                "--agents",
                "greedy",
                "random",
                "--shapes",
                "3x3",
                "--ks",
                "9",
                "--games",
                "2",
                "--quiet",
            ]
        )
        assert code == 2
        assert "no valid variants" in capsys.readouterr().err

    def test_bench_quick(self, capsys) -> None:
        assert main(["bench", "--preset", "tiny", "--quick"]) == 0
        assert "moves/s" in capsys.readouterr().out


class TestErrorHandling:
    def test_invalid_variant_exits_two(self, capsys) -> None:
        code = main(
            ["match", "--shape", "6", "7", "--k", "40", "--agents", "random", "random"]
        )
        assert code == 2
        assert "error:" in capsys.readouterr().err

    def test_unknown_agent_exits_two(self, capsys) -> None:
        code = main(
            ["match", "--preset", "tiny", "--agents", "nope", "random", "--games", "2"]
        )
        assert code == 2
        assert "error:" in capsys.readouterr().err


class TestSurfaceCommand:
    def test_surface(self, capsys) -> None:
        code = main(
            [
                "surface",
                "--agents",
                "random",
                "greedy",
                "--shapes",
                "4x5",
                "--ks",
                "3",
                "--games",
                "4",
                "--quiet",
            ]
        )
        assert code == 0
        output = capsys.readouterr().out
        assert "4x5k3p2" in output
        assert "spread" in output

    def test_surface_writes_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "surface.jsonl"
            main(
                [
                    "surface",
                    "--agents",
                    "random",
                    "greedy",
                    "--shapes",
                    "4x5",
                    "--ks",
                    "3",
                    "--games",
                    "4",
                    "--quiet",
                    "--jsonl",
                    str(path),
                ]
            )
            lines = path.read_text().strip().splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["variant"] == "4x5k3p2"

    def test_surface_with_no_valid_variants(self, capsys) -> None:
        code = main(
            [
                "surface",
                "--agents",
                "random",
                "greedy",
                "--shapes",
                "3x3",
                "--ks",
                "9",
                "--games",
                "2",
                "--quiet",
            ]
        )
        assert code == 2
        assert "no valid variants" in capsys.readouterr().err
