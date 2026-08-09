"""Enumerating variants and sweeping agents across them.

This is the module the project's thesis actually cashes out in. A single
match tells you which agent is stronger on one board; a sweep tells you whether
that ordering *survives* a change of board shape, win length, or player count —
which is the question generalization research is asking.

Invalid combinations are skipped rather than raised, so a grid can be specified
loosely (``k`` up to 6 across boards from 4x4 to 12x12) and the unwinnable
corners simply drop out.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from connectx.arena import (
    EngineFactory,
    MatchResult,
    TournamentResult,
    play_match,
    round_robin,
)
from connectx.config import ConfigError, config_id, make_config
from connectx.game import Game
from connectx.results import append_jsonl, completed_variants, provenance, read_jsonl
from connectx.types import Config

__all__ = [
    "variant_grid",
    "SweepResult",
    "sweep",
    "TournamentSurface",
    "tournament_surface",
]

PathLike = str | Path


def variant_grid(
    shapes: Iterable[Sequence[int]],
    ks: Iterable[int],
    player_counts: Iterable[int] = (2,),
) -> list[Config]:
    """Cross shapes, win lengths, and player counts into a list of valid configs.

    Combinations that cannot describe a playable game — ``k`` longer than the
    board's longest axis, for instance — are dropped silently, which is what
    makes a coarse specification usable.
    """
    configs: list[Config] = []
    for shape in shapes:
        for k in ks:
            for count in player_counts:
                try:
                    configs.append(
                        make_config(
                            (int(shape[0]), int(shape[1])),
                            int(k),
                            list(range(1, int(count) + 1)),
                        )
                    )
                except ConfigError:
                    continue
    return configs


@dataclass
class SweepResult:
    """One :class:`~connectx.arena.MatchResult` per variant, plus reporting."""

    specs: list[str]
    matches: list[MatchResult] = field(default_factory=list)
    #: Variants left unplayed because their seat count did not match ``specs``.
    skipped: list[str] = field(default_factory=list)

    @property
    def variants(self) -> list[str]:
        return [config_id(match.config) for match in self.matches]

    def score_rates(self, entrant: int = 0) -> dict[str, float]:
        """Score rate per variant for one entrant — the generalization curve."""
        return {
            config_id(match.config): match.score_rate(entrant) for match in self.matches
        }

    def table(self, entrant: int = 0) -> str:
        if not self.matches:
            return "(no variants)"
        name = self.specs[entrant]
        width = max(len(config_id(m.config)) for m in self.matches)
        header = (
            f"{'variant':<{width}}  {'score':>7}  {'95% CI':>16}  "
            f"{'draws':>6}  {'steps':>7}  {'moves/s':>9}"
        )
        lines = [f"{name} across {len(self.matches)} variants", header, "-" * len(header)]
        for match in self.matches:
            low, high = match.interval(entrant)
            lines.append(
                f"{config_id(match.config):<{width}}  "
                f"{match.score_rate(entrant):6.1%}  "
                f"[{low:5.1%}, {high:5.1%}]  "
                f"{match.draws:6d}  {match.mean_steps:7.1f}  "
                f"{match.moves_per_second:9,.0f}"
            )
        rates = list(self.score_rates(entrant).values())
        if rates:
            lines.append("-" * len(header))
            lines.append(
                f"{'mean':<{width}}  {sum(rates) / len(rates):6.1%}   "
                f"min {min(rates):.1%}  max {max(rates):.1%}  "
                f"spread {max(rates) - min(rates):.1%}"
            )
        return "\n".join(lines)

    def to_records(self) -> list[dict[str, Any]]:
        return [match.to_dict() for match in self.matches]

    def write_jsonl(self, filepath: PathLike) -> Path:
        """Append one JSON object per variant — the format experiments log in."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as handle:
            for record in self.to_records():
                handle.write(json.dumps(record) + "\n")
        return path


def sweep(
    specs: Sequence[str],
    configs: Sequence[Config],
    *,
    games: int = 50,
    seed: int | None = 0,
    workers: int = 1,
    engine: EngineFactory = Game,
    on_variant: Callable[[MatchResult], None] | None = None,
    jsonl: PathLike | None = None,
    resume: bool = False,
) -> SweepResult:
    """Run the same matchup on every variant.

    Each variant gets its own derived seed, so adding or reordering variants
    does not perturb the others' results. Variants whose seat count differs
    from ``len(specs)`` are recorded in :attr:`SweepResult.skipped` rather than
    raising, so a loosely specified grid stays runnable.

    Pass ``jsonl`` to append each variant's record the moment it finishes, so a
    run that dies partway keeps everything already computed. Add ``resume=True``
    to skip variants already present in that file and load their results back.
    """
    matches: list[MatchResult] = []
    skipped: list[str] = []
    done: set[str] = set()
    cached: dict[str, MatchResult] = {}
    if jsonl is not None and resume:
        done = completed_variants(jsonl)
        cached = {
            str(record["variant"]): MatchResult.from_dict(record)
            for record in read_jsonl(jsonl)
            if "variant" in record and "wins" in record
        }

    for index, config in enumerate(configs):
        variant = config_id(config)
        if len(config["players"]) != len(specs):
            skipped.append(variant)
            continue
        if variant in done:
            if variant in cached:
                matches.append(cached[variant])
            continue
        match = play_match(
            config,
            specs,
            games=games,
            seed=None if seed is None else seed + index * 7919,
            workers=workers,
            engine=engine,
        )
        matches.append(match)
        if jsonl is not None:
            append_jsonl(jsonl, match.to_dict())
        if on_variant is not None:
            on_variant(match)
    return SweepResult(specs=list(specs), matches=matches, skipped=skipped)


@dataclass
class TournamentSurface:
    """A full round robin run on every variant — the generalization surface.

    Each variant's ratings are fitted independently and anchored to the same
    mean, so **only differences within a column are meaningful**. Comparing an
    agent's absolute Elo across variants says nothing; comparing the *gaps*, the
    spread, and the ordering does. That is deliberate: an agent cannot be
    "1700 Elo" in the abstract, only 1700 relative to the field it played.
    """

    specs: list[str]
    tournaments: list[TournamentResult] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    @property
    def variants(self) -> list[str]:
        return [config_id(t.config) for t in self.tournaments]

    def rating_matrix(self) -> dict[str, dict[str, float]]:
        """``{agent: {variant: elo}}``."""
        return {
            spec: {
                config_id(t.config): t.ratings.get(spec, float("nan"))
                for t in self.tournaments
            }
            for spec in self.specs
        }

    def spread(self) -> dict[str, float]:
        """Elo range between best and worst agent, per variant.

        A small spread means the variant fails to separate these agents — the
        board is too shallow for skill to express itself.
        """
        out = {}
        for tournament in self.tournaments:
            values = list(tournament.ratings.values())
            out[config_id(tournament.config)] = (
                max(values) - min(values) if values else 0.0
            )
        return out

    def ranks(self) -> dict[str, dict[str, int]]:
        """``{agent: {variant: rank}}``, 1 = strongest on that variant."""
        out: dict[str, dict[str, int]] = {spec: {} for spec in self.specs}
        for tournament in self.tournaments:
            variant = config_id(tournament.config)
            order = sorted(tournament.ratings, key=lambda s: -tournament.ratings[s])
            for position, spec in enumerate(order, start=1):
                out[spec][variant] = position
        return out

    def rank_changes(self) -> list[str]:
        """Agents whose position in the ladder is not the same on every variant.

        These are the interesting rows: a reordering means the comparison does
        not transfer, which is exactly what a single-variant benchmark hides.
        """
        ranks = self.ranks()
        return [spec for spec in self.specs if len(set(ranks[spec].values())) > 1]

    def table(self) -> str:
        if not self.tournaments:
            return "(no variants)"
        variants = self.variants
        matrix = self.rating_matrix()
        ranks = self.ranks()
        # Strongest-on-average first, purely for readability.
        order = sorted(
            self.specs,
            key=lambda s: -sum(matrix[s].values()) / max(len(variants), 1),
        )
        rank_label = f"rank of {order[0]}"
        width = max(max(len(s) for s in self.specs), len(rank_label))
        columns = max(max(len(v) for v in variants), 8)

        header = f"{'agent':<{width}}  " + "  ".join(f"{v:>{columns}}" for v in variants)
        lines = [
            "Elo by variant (each column fitted independently, mean 1500;",
            "compare within a column, never across)",
            "",
            header,
            "-" * len(header),
        ]
        for spec in order:
            cells = "  ".join(f"{matrix[spec][v]:>{columns},.0f}" for v in variants)
            lines.append(f"{spec:<{width}}  {cells}")

        lines.append("-" * len(header))
        spreads = self.spread()
        lines.append(
            f"{'spread':<{width}}  "
            + "  ".join(f"{spreads[v]:>{columns},.0f}" for v in variants)
        )
        lines.append(
            f"{rank_label:<{width}}  "
            + "  ".join(f"{ranks[order[0]][v]:>{columns}}" for v in variants)
        )

        moved = self.rank_changes()
        lines.append("")
        if moved:
            lines.append(
                f"ladder order is NOT stable across variants; moved: {', '.join(moved)}"
            )
        else:
            lines.append("ladder order is identical on every variant")
        return "\n".join(lines)

    def to_records(self) -> list[dict[str, Any]]:
        # Same writer the streaming path uses, so a resumed file and a
        # written-at-the-end file are byte-identical in shape.
        return [_surface_record(t, self.specs) for t in self.tournaments]

    def write_jsonl(self, filepath: PathLike) -> Path:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as handle:
            for record in self.to_records():
                handle.write(json.dumps(record) + "\n")
        return path


def tournament_surface(
    specs: Sequence[str],
    configs: Sequence[Config],
    *,
    games: int = 40,
    seed: int | None = 0,
    workers: int = 1,
    engine: EngineFactory = Game,
    on_variant: Callable[[TournamentResult], None] | None = None,
    jsonl: PathLike | None = None,
    resume: bool = False,
) -> TournamentSurface:
    """Run a full round robin on every variant.

    Two-player variants only, since a round robin is a pairwise construction;
    anything else is recorded in :attr:`TournamentSurface.skipped`.

    Supports the same incremental ``jsonl`` writing and ``resume`` as
    :func:`sweep`. Resumed variants carry their ratings but not their underlying
    matches, which is enough for the table and the records.
    """
    tournaments: list[TournamentResult] = []
    skipped: list[str] = []
    done: set[str] = set()
    cached: dict[str, TournamentResult] = {}
    if jsonl is not None and resume:
        done = completed_variants(jsonl)
        for record in read_jsonl(jsonl):
            if "variant" not in record or "ratings" not in record:
                continue
            raw = record["config"]
            cached[str(record["variant"])] = TournamentResult(
                config=make_config(tuple(raw["shape"]), raw["k"], raw["players"]),
                specs=list(specs),
                matches={},
                ratings=dict(record["ratings"]),
                seed=record.get("provenance", {}).get("seed"),
                provenance=record.get("provenance", {}),
            )

    for index, config in enumerate(configs):
        variant = config_id(config)
        if len(config["players"]) != 2:
            skipped.append(variant)
            continue
        if variant in done:
            if variant in cached:
                tournaments.append(cached[variant])
            continue
        tournament = round_robin(
            config,
            specs,
            games=games,
            seed=None if seed is None else seed + index * 7919,
            workers=workers,
            engine=engine,
        )
        tournaments.append(tournament)
        if jsonl is not None:
            append_jsonl(jsonl, _surface_record(tournament, specs))
        if on_variant is not None:
            on_variant(tournament)
    return TournamentSurface(specs=list(specs), tournaments=tournaments, skipped=skipped)


def _surface_record(tournament: TournamentResult, specs: Sequence[str]) -> dict[str, Any]:
    """One variant's row of the surface, matching TournamentSurface.to_records."""
    order = sorted(tournament.ratings, key=lambda s: -tournament.ratings[s])
    values = list(tournament.ratings.values())
    return {
        "variant": config_id(tournament.config),
        "config": {
            "shape": list(tournament.config["shape"]),
            "k": tournament.config["k"],
            "players": list(tournament.config["players"]),
        },
        "ratings": dict(tournament.ratings),
        "ranks": {spec: order.index(spec) + 1 for spec in specs},
        "spread": (max(values) - min(values)) if values else 0.0,
        "provenance": tournament.provenance or provenance(tournament.seed),
    }
