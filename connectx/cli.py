"""Command line interface: ``connectx <command>`` (or ``python -m connectx``).

Every command takes the same variant flags, so the same invocation shape works
whether you are watching one game or sweeping fifty boards::

    connectx play    --preset connect4 --agents human minimax:depth=6
    connectx match   --shape 6 7 --k 4 --agents minimax:depth=4 greedy --games 200
    connectx tournament --preset connect4 --agents random greedy mcts minimax:depth=4
    connectx sweep   --agents minimax:depth=4 random --shapes 5x6 6x7 8x9 --ks 3 4
    connectx bench   --preset connect4
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from connectx.config import PRESETS, ConfigError, config_id, make_config, preset
from connectx.types import Config

__all__ = ["main", "build_parser"]


def _parse_shape(text: str) -> tuple[int, int]:
    """Accept ``6x7`` or ``6,7``."""
    cleaned = text.lower().replace(",", "x").strip()
    parts = [p for p in cleaned.split("x") if p]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"expected ROWSxCOLS, got {text!r}")
    try:
        return int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected ROWSxCOLS, got {text!r}") from exc


def _add_variant_flags(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("variant")
    group.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        help="named variant; overridden by --shape/--k/--players",
    )
    group.add_argument(
        "--shape", nargs=2, type=int, metavar=("ROWS", "COLS"), help="board size"
    )
    group.add_argument("--k", type=int, help="tokens in a row needed to win")
    group.add_argument(
        "--players", type=int, help="number of players (tokens 1..n)", default=None
    )


def _config_from_args(args: argparse.Namespace) -> Config:
    base = preset(args.preset) if args.preset else preset("connect4")
    shape = tuple(args.shape) if args.shape else base["shape"]
    k = args.k if args.k is not None else base["k"]
    if args.players is not None:
        players = list(range(1, args.players + 1))
    else:
        players = base["players"]
    return make_config(shape, k, players)


# ----------------------------------------------------------------------
# Commands
# ----------------------------------------------------------------------


def cmd_play(args: argparse.Namespace) -> int:
    from agents import agent_reset, make_agent
    from connectx import Game

    config = _config_from_args(args)
    if len(args.agents) != len(config["players"]):
        print(
            f"error: {len(config['players'])} seats but {len(args.agents)} agents given",
            file=sys.stderr,
        )
        return 2

    agents = [
        make_agent(spec, config, seed=None if args.seed is None else args.seed + i)
        for i, spec in enumerate(args.agents)
    ]
    for agent in agents:
        agent_reset(agent, config)

    game = Game(config)
    state, actions = game.start()
    while not game.terminal():
        if not args.quiet:
            game.render()
        seat = state["info"]["active"]
        action = agents[seat].select(state, actions)
        if not args.quiet:
            print(f"  {args.agents[seat]} -> column {action}")
        state, actions = game.transition(action)

    if not args.quiet:
        game.render()
    report = game.report()
    if report["winner"] is None:
        print("Draw.")
    else:
        print(
            f"Winner: {args.agents[report['winner']['id']]} (seat {report['winner']['id']})"
        )
    print(f"Moves: {report['steps']}")
    return 0


def cmd_match(args: argparse.Namespace) -> int:
    from connectx.arena import play_match

    config = _config_from_args(args)
    result = play_match(
        config,
        args.agents,
        games=args.games,
        swap_seats=not args.no_swap,
        seed=args.seed,
        workers=args.workers,
    )
    print(result.summary())
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(result.to_dict(), indent=2))
        print(f"\nwrote {args.json}")
    return 0


def cmd_tournament(args: argparse.Namespace) -> int:
    from connectx.arena import round_robin

    config = _config_from_args(args)
    result = round_robin(
        config,
        args.agents,
        games=args.games,
        seed=args.seed,
        workers=args.workers,
    )
    print(f"round robin on {config_id(config)}  ({args.games} games per pairing)\n")
    print(result.table())
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(result.to_dict(), indent=2))
        print(f"\nwrote {args.json}")
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    from connectx.arena import MatchResult
    from connectx.variants import sweep, variant_grid

    if args.shapes:
        shapes = [_parse_shape(s) for s in args.shapes]
    else:
        shapes = [(5, 6), (6, 7), (7, 8), (8, 9)]
    ks = args.ks or [3, 4, 5]
    counts = args.player_counts or [len(args.agents)]

    configs = variant_grid(shapes, ks, counts)
    if not configs:
        print("error: no valid variants in that grid", file=sys.stderr)
        return 2

    def progress(match: MatchResult) -> None:
        if not args.quiet:
            print(f"  {config_id(match.config):<12} done", flush=True)

    if not args.quiet:
        print(f"sweeping {len(configs)} variants x {args.games} games\n")
    result = sweep(
        args.agents,
        configs,
        games=args.games,
        seed=args.seed,
        workers=args.workers,
        on_variant=progress,
    )
    print()
    print(result.table())
    if result.skipped:
        print(f"\nskipped (seat count mismatch): {', '.join(result.skipped)}")
    if args.jsonl:
        path = result.write_jsonl(args.jsonl)
        print(f"\nwrote {path}")
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    from connectx.benchmark import run_benchmarks

    run_benchmarks(_config_from_args(args), quick=args.quick, json_path=args.json)
    return 0


def cmd_agents(args: argparse.Namespace) -> int:
    from agents import available_agents

    print("available agents (use name or name:key=value,key=value):")
    for name in available_agents():
        print(f"  {name}")
    print("\npresets:")
    for name, config in sorted(PRESETS.items()):
        print(f"  {name:<14} {config_id(config)}")
    return 0


# ----------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="connectx",
        description="Parameterized Connect-style games for training and evaluating agents.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    play = subparsers.add_parser("play", help="play a single game and render it")
    _add_variant_flags(play)
    play.add_argument("--agents", nargs="+", default=["random", "random"])
    play.add_argument("--seed", type=int, default=None)
    play.add_argument("--quiet", action="store_true", help="only print the result")
    play.set_defaults(func=cmd_play)

    match = subparsers.add_parser(
        "match", help="many games between fixed agents, with confidence intervals"
    )
    _add_variant_flags(match)
    match.add_argument("--agents", nargs="+", required=True)
    match.add_argument("--games", type=int, default=100)
    match.add_argument("--seed", type=int, default=0)
    match.add_argument("--workers", type=int, default=1)
    match.add_argument(
        "--no-swap", action="store_true", help="do not rotate seats (not recommended)"
    )
    match.add_argument("--json", help="write the result to this path")
    match.set_defaults(func=cmd_match)

    tournament = subparsers.add_parser("tournament", help="round robin with Elo ratings")
    _add_variant_flags(tournament)
    tournament.add_argument("--agents", nargs="+", required=True)
    tournament.add_argument("--games", type=int, default=50)
    tournament.add_argument("--seed", type=int, default=0)
    tournament.add_argument("--workers", type=int, default=1)
    tournament.add_argument("--json", help="write the result to this path")
    tournament.set_defaults(func=cmd_tournament)

    sweep_parser = subparsers.add_parser(
        "sweep", help="run one matchup across a grid of variants"
    )
    sweep_parser.add_argument("--agents", nargs="+", required=True)
    sweep_parser.add_argument(
        "--shapes", nargs="+", metavar="ROWSxCOLS", help="e.g. 5x6 6x7 8x9"
    )
    sweep_parser.add_argument("--ks", nargs="+", type=int)
    sweep_parser.add_argument(
        "--player-counts", nargs="+", type=int, dest="player_counts"
    )
    sweep_parser.add_argument("--games", type=int, default=50)
    sweep_parser.add_argument("--seed", type=int, default=0)
    sweep_parser.add_argument("--workers", type=int, default=1)
    sweep_parser.add_argument("--quiet", action="store_true")
    sweep_parser.add_argument("--jsonl", help="write one JSON record per variant")
    sweep_parser.set_defaults(func=cmd_sweep)

    bench = subparsers.add_parser("bench", help="engine throughput benchmarks")
    _add_variant_flags(bench)
    bench.add_argument("--quick", action="store_true", help="fewer iterations")
    bench.add_argument("--json", help="write results to this path")
    bench.set_defaults(func=cmd_bench)

    listing = subparsers.add_parser("agents", help="list agents and presets")
    listing.set_defaults(func=cmd_agents)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
