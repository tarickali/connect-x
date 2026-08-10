# Changelog

## Unreleased

### Added

- **The agent ladder plays any engine.** `GreedyAgent`, `MinimaxAgent`, and
  `MCTSAgent` called the drop primitives directly, so a new engine could pass
  the conformance suite and still have no baseline able to play it. They now
  search through `agents.search.Position`, an engine-agnostic push/pop cursor,
  and build their lookahead engine from an injected factory that `play_match`
  threads through. `Game.rollout` exposes the compiled playout as an optional
  capability; `Position` falls back to stepping the engine when an engine does
  not offer one.
- `tests/free_placement.py`, a free-placement m,n,k engine used as a fixture:
  no gravity, a `rows * cols` action space, no compiled rollout. The conformance
  suite runs the full contract and the whole agent ladder against it, so
  "works with any engine" is tested rather than claimed.
- A **solved-game oracle**: at 3x3 the free-placement engine is tic-tac-toe, and
  `minimax:depth=9` is checked to draw itself every game and never lose to
  anything. That is the project's first absolute measure of playing strength;
  everything else is agents against each other.
- `scripts/check_guide.py`, which re-derives the counts and module sizes the
  docs quote and fails CI when they drift.

- **Provenance on every result.** Records now carry the seed, a UTC timestamp,
  the git SHA and dirty flag, and the connectx/Python/numpy/numba versions. A
  record previously could not be reproduced from itself, because the seed that
  produced it was never written down. New module: `connectx.results`.
- **Sample-size planning.** `games_needed(elo_gap)` and `resolvable_gap(games)`,
  plus `elo_to_score` / `score_to_elo`. Applying these to the first surface run
  showed its small-board result was noise: 40 games resolves only ~160 Elo, and
  the reported gap was 42.
- **Resumable runs.** `sweep` and `surface` take `jsonl=` to append each
  variant's record as it completes and `resume=True` (CLI: `--resume`) to skip
  variants already present, so a run that dies partway keeps its work.
- **Engine factories.** `play_game`, `play_match`, `round_robin`, `sweep`,
  `tournament_surface`, both adapters, and the benchmarks all take an `engine=`
  argument instead of importing `Game`.
- `GameEngine.action_space_size`, so policy widths come from the engine rather
  than assuming one action per column. `build_supervised` uses it.
- `connectx.implements_engine` and `connectx.protocol_members`, the
  version-independent way to check an engine against the protocol.
- `tests/test_engine_conformance.py` — a dynamics-agnostic contract suite. Add a
  new game to `ENGINES` and it is checked against the protocol, the arena, and
  the dataset pipeline.

### Fixed

- **`Game.transition` discarded the seat a resumed state was started with.** It
  recomputed `active` as `time % n_players`, so a position where the two
  disagreed silently corrected itself on the next move — an agent searching from
  such a position had the wrong side to move. It now advances from the seat that
  actually moved. `VecGame` had the same rule and the same fix. Found by routing
  minimax through the engine, which the old direct-primitive search had masked.
- **`isinstance(x, GameEngine)` raised on Python 3.11 and earlier.** A
  runtime-checkable protocol check calls `hasattr` on every member, which
  evaluates properties; `Game.state` raises before `start()`, so the check blew
  up instead of returning a bool. Python 3.12 switched to
  `inspect.getattr_static` and was unaffected, which is why CI was red on 3.10
  and 3.11 only. Added `connectx.implements_engine`, which inspects the type
  statically and behaves the same on every version, and documented the pitfall
  on the protocol.
- `Trajectory.replay` rebuilt positions with the gravity-based `place_token`
  rather than replaying through the engine, so a game with different placement
  rules would have replayed *wrong* rather than failed — silently mislabelling
  anything `build_supervised` produced from it.
- `TournamentSurface.to_records` and the streaming writer emitted different
  record shapes; they now share one writer.
- Corrected the README's headline result. The 40-game surface reported the agent
  ladder reordering on small boards; at 600 games that difference is zero. The
  real finding is that `k=3` variants are forced first-player wins both agents
  convert, so they measure nothing about agent skill — visible in `seat_wins`
  reading 200-0.

### Added (earlier in this release)

- `connectx surface` and `connectx.variants.tournament_surface` — a full round
  robin on every variant, with per-variant Elo, ranks, and the spread between
  best and worst agent. Ratings are fitted per variant and anchored to the same
  mean, so only within-column differences are meaningful; the table says so, and
  reports explicitly whether the ladder order is stable across variants.
- `scripts/ablation.py` — measures whether numba earns its place, comparing the
  shipped JIT build against both the same source uncompiled and an idiomatic
  numpy implementation. Result: 14x on a 200-game match, 7.5x on batched
  environments, 145x on a full-board win scan, and a tie where the work is
  genuinely vectorizable. The whole suite passes under `NUMBA_DISABLE_JIT=1`, so
  numba is a pure accelerator, never load-bearing.

### Changed

- Search is about 2.7x slower now that it goes through the engine
  (`minimax:depth=4` 0.32 ms -> 0.86 ms per move). Node counts are unchanged, so
  the search itself is identical; the cost is `Game.transition` being
  Python-bound. Judged a fair price for a ladder that works on any game.
- Corrected a fourth documentation error: `games_needed(40)` is 596, not 874.
  The wrong figure came from `games_needed(33)` measured elsewhere.
- CI now covers Python 3.10 through 3.14 (3.14 was the development version but
  went untested), runs the suite a second time with `NUMBA_DISABLE_JIT=1` to
  prove the claim that numba is a pure accelerator, and uses action versions
  that run on Node 24.
- `MCTSAgent` is ~2x faster with identical move selection. Profiling showed only
  ~28% of a search was inside the compiled rollout; the rest was Python tree
  overhead. Backup vectors are now precomputed per outcome in `reset` instead of
  allocated per simulation, node statistics are Python floats rather than
  one-element numpy operations, and expansion reuses a single legality mask.
- Documented the numba trade-off and the generalization surface in the README.

## 0.2.0

Breaking release. The engine, agent interface, and result types all changed;
see [USAGE.md](USAGE.md) for the current API.

### Fixed

- **A win that also filled the board was reported as a draw.** `report()`
  consulted `check_tie` before the winner, and `terminal()` could not
  distinguish "someone connected `k`" from "no cells left". Outcomes now come
  from `winner()`, and a draw requires a full board *and* no line.
- **`MinimaxAgent` scored the opponent's pieces as its own at odd depths.**
  `evaluate` received the token of whoever moved last rather than the
  minimizing player, so at depths 1, 3, 5, ... the maximizing and minimizing
  tokens were identical. The default depth was one of the broken ones. The
  search is now negamax, which has no such pair of arguments to desynchronize.
- **Board dimensions and `k` wrapped silently at 255.** `Shape` and the numba
  signatures typed them as `uint8`, so `create_grid((300, 7))` returned a
  `(44, 7)` board. Now `int64` throughout, with no upper bound.
- **Nothing validated a config.** `k=9` on a 6x7 board (unwinnable), token `0`
  (invisible on the board, so that player's columns never filled), duplicate
  tokens, a single player, and zero-sized boards were all accepted. See
  `connectx.config.validate_config`.
- **`Game.state` handed out live internal objects.** Callers could edit the
  board or the clock through the state they were given. The grid is now a
  non-writeable view and the turn counters are copied.
- **`RandomAgent` used the global numpy RNG**, so nothing was reproducible and
  parallel workers shared state. Every agent now owns a seeded `Generator`.
- **`place_token` silently dropped moves on a full column** with no signal to
  the functional API. Added `drop_row` and `is_legal`; `Game.transition` raises.
- Fixed `requirements.txt` pinning `numpy==1.26` against a `pyproject.toml`
  range that `numba` could not satisfy.
- Fixed `scripts/benchmark.py` failing to import from a clean checkout.
- Replaced a test that asserted `len(set(choices)) >= 1`, which is true of any
  non-empty list.

### Changed

- `terminal()` is O(1) after an O(k) incremental update, instead of rescanning
  the board. On 60x60 this took win detection from 330 µs to 0.15 µs.
- `MinimaxAgent` gained alpha-beta pruning, centre-first move ordering, a
  transposition table, depth-discounted mate scores, and optional iterative
  deepening under a time budget. A depth-5 opening move went from 6.0 s to
  0.002 s; depth 8 now costs less than depth 4 did before.
- `Game.step(action)` returns `StepResult(state, actions, rewards, terminated,
  report)`. `transition(action)` keeps the `(state, actions)` shape.
- `Game` rejects moves after the game has ended.
- `Trajectory` records the outcome and exposes `returns()` (per-seat value
  targets) and `replay()` (positions regenerated from the move list).
- Serialization moved from `pickle` to `npz` + JSON.
- `Agent` gained lifecycle hooks (`reset(config)`, `observe`, `seed`) via
  `BaseAgent`. Agents derive variant state in `reset`, so one instance can play
  any variant.
- `main.py` no longer duplicates `recipes/class_example.py`; it forwards to the
  CLI. `notebook.ipynb` moved to `notebooks/prototype.ipynb`.
- The renderer prints symbols and marks the winning line.
- `check_tie(grid)` → `is_draw(grid, k)` and `full(grid)`. The internal
  `check_horizontal_lines` / `check_vertical_lines` / `check_diagonal_lines`
  helpers are replaced by `winner`, `winner_at`, and `winner_line`.
- `Action` is now `int` (a column index) rather than `np.uint8`.

### Added

- `connectx.config` — validation, `config_id`, and eight named presets.
- `connectx.vector.VecGame` — batched environments, 6.5M moves/second at 2048.
- `connectx.encoding` — perspective-relative planes, action masks, mirror
  symmetry.
- `connectx.dataset` — trajectory shards and `build_supervised`.
- `connectx.arena` — matches with seat rotation and Wilson intervals,
  round-robin tournaments with Bradley-Terry Elo, process parallelism with
  index-derived seeds so worker count never changes a result.
- `connectx.variants` — variant grids and sweeps, the generalization measurement.
- `connectx.adapters` — PettingZoo AEC and Gymnasium environments, both passing
  upstream conformance tests.
- `connectx.benchmark` and `connectx.cli` — `connectx play | match | tournament
  | sweep | bench | agents`, also available as `python -m connectx`.
- `GreedyAgent`, `MCTSAgent` (any number of players), `HumanAgent`, and a spec
  registry: `make_agent("mcts:simulations=800")`.
- `RewardSpec`, per-seat `rewards()`, and populated `TrajectoryStep.reward`.
- `Game.undo` restores terminal status; `fork`, `instance`, `from_instance`.
- `scripts/experiment.py` and JSON experiment specs.
- CI (tests on 3.10-3.13, ruff, mypy, and a smoke test of every documented
  command), `py.typed`, and USAGE.md.
- Test suite grew from 40 to 275 tests.

## 0.1.0

Initial release: functional and class APIs, random and minimax agents,
multiplayer support, benchmark script, packaging metadata.
