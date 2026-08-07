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

from connectx.arena import MatchResult, play_match
from connectx.config import ConfigError, config_id, make_config
from connectx.types import Config

__all__ = [
    "variant_grid",
    "SweepResult",
    "sweep",
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
    on_variant: Callable[[MatchResult], None] | None = None,
) -> SweepResult:
    """Run the same matchup on every variant.

    Each variant gets its own derived seed, so adding or reordering variants
    does not perturb the others' results. Variants whose seat count differs
    from ``len(specs)`` are recorded in :attr:`SweepResult.skipped` rather than
    raising, so a loosely specified grid stays runnable.
    """
    matches: list[MatchResult] = []
    skipped: list[str] = []
    for index, config in enumerate(configs):
        if len(config["players"]) != len(specs):
            skipped.append(config_id(config))
            continue
        match = play_match(
            config,
            specs,
            games=games,
            seed=None if seed is None else seed + index * 7919,
            workers=workers,
        )
        matches.append(match)
        if on_variant is not None:
            on_variant(match)
    return SweepResult(specs=list(specs), matches=matches, skipped=skipped)
