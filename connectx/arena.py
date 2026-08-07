"""Measurement: matches, tournaments, ratings, and the statistics to read them.

Three things here exist because leaving them out is how agent comparisons go
wrong:

- **Seat rotation.** Connect-style games are strongly first-player biased, so
  an unbalanced pairing measures the seat, not the agent. Every match plays
  each ordering an equal number of times and reports the per-seat split so the
  bias is visible rather than hidden.
- **Confidence intervals.** A win rate without an interval cannot be acted on.
  Wilson intervals are used because they stay sensible at the 0% and 100% ends,
  which is exactly where agent comparisons land.
- **Seed derivation.** Every game's seed is a pure function of the match seed,
  the game index, and the seat. Results are therefore identical whether the
  match runs on one worker or twelve.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from connectx.config import config_id, validate_config
from connectx.game import Game
from connectx.types import Config

__all__ = [
    "GameResult",
    "MatchResult",
    "TournamentResult",
    "wilson_interval",
    "play_game",
    "play_match",
    "round_robin",
    "elo_ratings",
]


# ----------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------


def wilson_interval(successes: float, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Preferred over the normal approximation because it does not produce
    impossible bounds when an agent wins every game — the common case when
    something on the ladder is simply stronger.
    """
    if total <= 0:
        return (0.0, 1.0)
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    spread = (
        z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    )
    return (max(0.0, center - spread), min(1.0, center + spread))


# ----------------------------------------------------------------------
# Single game
# ----------------------------------------------------------------------


@dataclass
class GameResult:
    """Outcome of one game, with results attributed to entrants, not seats."""

    winner_seat: int | None
    winner_entrant: int | None
    tie: bool
    steps: int
    duration: float
    seating: list[int]  # seat -> entrant index

    def to_dict(self) -> dict[str, Any]:
        return {
            "winner_seat": self.winner_seat,
            "winner_entrant": self.winner_entrant,
            "tie": self.tie,
            "steps": self.steps,
            "duration": self.duration,
            "seating": list(self.seating),
        }


def _derive_seed(base: int | None, game_index: int, seat: int) -> int | None:
    """Deterministic per-agent seed, so worker count never changes a result."""
    if base is None:
        return None
    return int((base * 1_000_003 + game_index * 9_176 + seat * 31) % (2**31 - 1))


def play_game(
    config: Config,
    specs: Sequence[str],
    *,
    seating: Sequence[int] | None = None,
    seed: int | None = None,
    game_index: int = 0,
    record: bool = False,
) -> tuple[GameResult, Any | None]:
    """Play one game between agents built from ``specs``.

    ``seating[seat] = entrant`` maps board seats to entrants, which is how seat
    rotation is expressed. Returns the result and, with ``record=True``, the
    trajectory.
    """
    from agents import agent_observe, agent_reset, make_agent

    validate_config(config)
    n_players = len(config["players"])
    if len(specs) != n_players:
        raise ValueError(
            f"need one agent spec per player: got {len(specs)} for {n_players} seats"
        )
    order = list(range(n_players)) if seating is None else list(seating)

    agents = []
    for seat in range(n_players):
        agent = make_agent(
            specs[order[seat]], config, seed=_derive_seed(seed, game_index, seat)
        )
        agent_reset(agent, config)
        agents.append(agent)

    game = Game(config, record=record)
    state, actions = game.start()

    started = time.perf_counter()
    while not game.terminal():
        seat = state["info"]["active"]
        action = agents[seat].select(state, actions)
        state, actions = game.transition(action)
    duration = time.perf_counter() - started

    payoffs = game.rewards()
    for seat, agent in enumerate(agents):
        agent_observe(agent, state, float(payoffs[seat]), True)

    report = game.report()
    winner_seat = None if report["winner"] is None else int(report["winner"]["id"])
    result = GameResult(
        winner_seat=winner_seat,
        winner_entrant=None if winner_seat is None else order[winner_seat],
        tie=bool(report["tie"]),
        steps=int(report["steps"]),
        duration=duration,
        seating=order,
    )
    return result, (game.trajectory() if record else None)


# ----------------------------------------------------------------------
# Match
# ----------------------------------------------------------------------


@dataclass
class MatchResult:
    """Aggregate of many games between the same entrants."""

    config: Config
    specs: list[str]
    games: int
    wins: list[int]
    draws: int
    seat_wins: list[int]
    total_steps: int
    duration: float

    def score(self, entrant: int) -> float:
        """Wins plus half the draws — the quantity Elo is fitted to."""
        return self.wins[entrant] + 0.5 * self.draws

    def win_rate(self, entrant: int) -> float:
        return self.wins[entrant] / self.games if self.games else 0.0

    def score_rate(self, entrant: int) -> float:
        return self.score(entrant) / self.games if self.games else 0.0

    def interval(self, entrant: int, z: float = 1.96) -> tuple[float, float]:
        """Wilson interval on the score rate."""
        return wilson_interval(self.score(entrant), self.games, z)

    @property
    def mean_steps(self) -> float:
        return self.total_steps / self.games if self.games else 0.0

    @property
    def moves_per_second(self) -> float:
        return self.total_steps / self.duration if self.duration else 0.0

    def summary(self) -> str:
        lines = [
            f"{config_id(self.config)}  {self.games} games  "
            f"({self.duration:.2f}s, {self.moves_per_second:,.0f} moves/s)"
        ]
        for index, spec in enumerate(self.specs):
            low, high = self.interval(index)
            lines.append(
                f"  {spec:<28} wins {self.wins[index]:>4}  "
                f"score {self.score_rate(index):6.1%}  "
                f"95% CI [{low:5.1%}, {high:5.1%}]"
            )
        lines.append(f"  {'draws':<28} {self.draws:>9}")
        lines.append(
            "  seat wins: "
            + ", ".join(f"seat {i}: {w}" for i, w in enumerate(self.seat_wins))
        )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant": config_id(self.config),
            "config": {
                "shape": list(self.config["shape"]),
                "k": self.config["k"],
                "players": list(self.config["players"]),
            },
            "specs": list(self.specs),
            "games": self.games,
            "wins": list(self.wins),
            "draws": self.draws,
            "seat_wins": list(self.seat_wins),
            "score_rate": [self.score_rate(i) for i in range(len(self.specs))],
            "interval": [list(self.interval(i)) for i in range(len(self.specs))],
            "mean_steps": self.mean_steps,
            "moves_per_second": self.moves_per_second,
            "duration": self.duration,
        }


def _seating_for(game_index: int, n_players: int) -> list[int]:
    """Rotate entrants through seats so every ordering is played equally often."""
    shift = game_index % n_players
    return [(seat + shift) % n_players for seat in range(n_players)]


def _play_chunk(
    config: Config,
    specs: Sequence[str],
    indices: Sequence[int],
    seed: int | None,
    swap_seats: bool,
) -> list[GameResult]:
    n_players = len(config["players"])
    results = []
    for index in indices:
        seating = _seating_for(index, n_players) if swap_seats else None
        result, _ = play_game(config, specs, seating=seating, seed=seed, game_index=index)
        results.append(result)
    return results


def play_match(
    config: Config,
    specs: Sequence[str],
    *,
    games: int = 100,
    swap_seats: bool = True,
    seed: int | None = 0,
    workers: int = 1,
) -> MatchResult:
    """Play ``games`` games between ``specs`` and aggregate the outcome.

    ``workers > 1`` distributes games across processes. Because seeds derive
    from the game index, the aggregate is bit-identical to a single-worker run.
    """
    validate_config(config)
    n_players = len(config["players"])
    if len(specs) != n_players:
        raise ValueError(
            f"need one agent spec per player: got {len(specs)} for {n_players} seats"
        )
    if games < 1:
        raise ValueError(f"games must be positive, got {games}")
    if swap_seats and games % n_players:
        # Otherwise one entrant gets an extra turn in the strongest seat.
        raise ValueError(
            f"games ({games}) must be a multiple of the player count ({n_players}) "
            "when swap_seats is on, so seat assignment stays balanced"
        )

    indices = list(range(games))
    started = time.perf_counter()
    if workers <= 1:
        results = _play_chunk(config, specs, indices, seed, swap_seats)
    else:
        chunks = [indices[i::workers] for i in range(workers)]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(_play_chunk, config, specs, chunk, seed, swap_seats)
                for chunk in chunks
                if chunk
            ]
            results = [game for future in futures for game in future.result()]
    duration = time.perf_counter() - started

    wins = [0] * len(specs)
    seat_wins = [0] * n_players
    draws = 0
    total_steps = 0
    for result in results:
        total_steps += result.steps
        if result.winner_entrant is None or result.winner_seat is None:
            draws += 1
        else:
            wins[result.winner_entrant] += 1
            seat_wins[result.winner_seat] += 1

    return MatchResult(
        config=config,
        specs=list(specs),
        games=len(results),
        wins=wins,
        draws=draws,
        seat_wins=seat_wins,
        total_steps=total_steps,
        duration=duration,
    )


# ----------------------------------------------------------------------
# Tournament
# ----------------------------------------------------------------------


def elo_ratings(
    specs: Sequence[str],
    scores: dict[tuple[int, int], tuple[float, float]],
    *,
    anchor: float = 1500.0,
    prior_games: float = 2.0,
    iterations: int = 1000,
    tolerance: float = 1e-10,
) -> dict[str, float]:
    """Fit Bradley-Terry strengths to pairwise scores and report them as Elo.

    ``scores[(i, j)] = (score_i, games_played)``. Fitted by the standard
    minorization-maximization iteration, with ``prior_games`` virtual draws
    against an average opponent so that an entrant who wins nothing (or
    everything) still receives a finite rating instead of diverging.

    Only rating *differences* carry meaning; the scale is pinned so the mean
    rating equals ``anchor``.
    """
    n = len(specs)
    if n == 0:
        return {}

    wins = [prior_games / 2.0] * n
    pairings = [[0.0] * n for _ in range(n)]
    for (i, j), (score_i, played) in scores.items():
        if played <= 0:
            continue
        pairings[i][j] += played
        pairings[j][i] += played
        wins[i] += score_i
        wins[j] += played - score_i

    strengths = [1.0] * n
    for _ in range(iterations):
        updated = list(strengths)
        largest_change = 0.0
        for i in range(n):
            # Virtual opponent of average strength keeps the update well-posed.
            denominator = prior_games / (strengths[i] + 1.0)
            for j in range(n):
                if i == j or pairings[i][j] <= 0:
                    continue
                denominator += pairings[i][j] / (strengths[i] + strengths[j])
            if denominator > 0:
                candidate = wins[i] / denominator
                largest_change = max(largest_change, abs(candidate - strengths[i]))
                updated[i] = candidate

        # Renormalize to a geometric mean of 1 so the scale cannot drift.
        geometric_mean = math.exp(sum(math.log(s) for s in updated) / n)
        strengths = [s / geometric_mean for s in updated]
        if largest_change < tolerance:
            break

    return {
        spec: anchor + 400.0 * math.log10(strengths[i]) for i, spec in enumerate(specs)
    }


@dataclass
class TournamentResult:
    config: Config
    specs: list[str]
    matches: dict[tuple[int, int], MatchResult] = field(default_factory=dict)
    ratings: dict[str, float] = field(default_factory=dict)

    def table(self) -> str:
        order = sorted(self.specs, key=lambda s: -self.ratings.get(s, 0.0))
        width = max((len(s) for s in self.specs), default=10)
        lines = [f"{'agent':<{width}}  {'elo':>7}  {'score':>7}  {'games':>6}"]
        lines.append("-" * len(lines[0]))
        for spec in order:
            index = self.specs.index(spec)
            score = 0.0
            games = 0
            for (i, j), match in self.matches.items():
                if i == index:
                    score += match.score(0)
                    games += match.games
                elif j == index:
                    score += match.score(1)
                    games += match.games
            rate = score / games if games else 0.0
            lines.append(
                f"{spec:<{width}}  {self.ratings.get(spec, 0.0):7.0f}  "
                f"{rate:6.1%}  {games:6d}"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant": config_id(self.config),
            "specs": list(self.specs),
            "ratings": dict(self.ratings),
            "matches": [
                {"pair": [i, j], **match.to_dict()}
                for (i, j), match in self.matches.items()
            ],
        }


def round_robin(
    config: Config,
    specs: Sequence[str],
    *,
    games: int = 50,
    seed: int | None = 0,
    workers: int = 1,
) -> TournamentResult:
    """Play every pair of entrants and fit Elo ratings to the results.

    Two-player variants only — a round robin is a pairwise construction.
    """
    validate_config(config)
    if len(config["players"]) != 2:
        raise ValueError(
            "round_robin pairs entrants two at a time; "
            f"this variant seats {len(config['players'])} players"
        )
    entrants = list(specs)
    if len(entrants) < 2:
        raise ValueError("need at least two agents for a round robin")

    matches: dict[tuple[int, int], MatchResult] = {}
    scores: dict[tuple[int, int], tuple[float, float]] = {}
    for i in range(len(entrants)):
        for j in range(i + 1, len(entrants)):
            match = play_match(
                config,
                [entrants[i], entrants[j]],
                games=games,
                seed=None if seed is None else seed + i * 97 + j,
                workers=workers,
            )
            matches[(i, j)] = match
            scores[(i, j)] = (match.score(0), float(match.games))

    return TournamentResult(
        config=config,
        specs=entrants,
        matches=matches,
        ratings=elo_ratings(entrants, scores),
    )
