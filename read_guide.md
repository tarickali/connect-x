# Reading guide

A review guide for the whole project, written to be read before you build on it.
It is ordered so that each part makes sense given what came before, and it is
explicit about what I decided, what I got wrong, and what you should distrust.

- [Is it ready?](#is-it-ready)
- [How to verify anything here yourself](#how-to-verify-anything-here-yourself)
- [Reading order](#reading-order)
- [The bugs that were fixed](#the-bugs-that-were-fixed)
- [Decisions you may want to overturn](#decisions-you-may-want-to-overturn)
- [Soft spots and untested corners](#soft-spots-and-untested-corners)
- [Claims made in the docs, and their evidence](#claims-made-in-the-docs-and-their-evidence)
- [Commit-by-commit history](#commit-by-commit-history)
- [Review checklist](#review-checklist)
- [Keeping this guide honest](#keeping-this-guide-honest)

---

## Is it ready?

**For agent design and experiments on the current variant family: yes.** The
engine is correct as far as the tests reach, fast enough that experiments are
cheap, reproducible from a seed, and instrumented well enough that you can tell
a real effect from noise. The measurement layer already produced one non-obvious
finding and already caught one false claim of mine.

**For research across *different game dynamics*: not yet, and deliberately so.**
Every variant today is one game with three scalar knobs (shape, `k`, players).
Gomoku and the other m,n,k games need a free-placement engine, which is
[section 2 of `todo.md`](todo.md). The seam they plug into is finished and
tested; the games themselves are not written.

Two things to decide early, because they are expensive to change later:

1. **PyTorch or JAX** for the neural agent. Keep it behind an `[nn]` extra.
2. Whether you accept the **hand-tuned heuristic weights** in
   `agents/heuristics.py` — see [soft spots](#soft-spots-and-untested-corners).
   Minimax's playing strength rests on numbers nobody fitted.

---

## How to verify anything here yourself

Do not take this document's word for anything. Everything below is checkable:

```bash
pip install -e ".[dev]"

pytest                                   # 513 tests, ~2 s
pytest --cov --cov-report=term-missing   # see the coverage caveat below
ruff check connectx agents tests recipes scripts
ruff format --check connectx agents tests recipes scripts
mypy                                     # 30 source files, clean

NUMBA_DISABLE_JIT=1 pytest               # all 403 still pass; numba is a pure accelerator
```

**Coverage caveat, read this before you look at the numbers.** `coverage.py`
cannot see inside numba-compiled functions, so a plain run reports
`functional.py` at 25% and `heuristics.py` at 26%, which is wrong. Disable the
JIT and the same code becomes visible:

| | plain run | `NUMBA_DISABLE_JIT=1` |
| --- | ---: | ---: |
| `connectx/functional.py` | 25% | **98%** |
| `agents/heuristics.py` | 26% | **100%** |
| `agents/rollout.py` | 18% | **98%** |
| `connectx/vector.py` | 80% | **99%** |
| **TOTAL** | 86% | **95%** |

So: `NUMBA_DISABLE_JIT=1 pytest --cov` is the honest coverage command.

---

## Reading order

Sizes are non-blank source lines. Read top to bottom; each layer only depends on
the ones above it.

### 1. The contracts — start here (~150 lines)

| File | Lines | Why first |
| --- | ---: | --- |
| `connectx/types.py` | 76 | Every other module speaks these types. `Config`, `State`, `Report`, `RewardSpec`, `StepResult`. |
| `connectx/engine.py` | 105 | `GameEngine`, the protocol a game must satisfy. Read the class docstring for the factory contract. |

Read `types.py` closely. `Info["active"]` is a **seat index**, not a token;
`config["players"][seat]` is the token. Almost every off-by-one risk in the
codebase is that distinction.

### 2. The engine (~600 lines)

| File | Lines | What to look at |
| --- | ---: | --- |
| `connectx/config.py` | 90 | `validate_config` — the invariants everything else assumes. |
| `connectx/functional.py` | 200 | Pure JIT primitives. Read `winner_at` and `winner_line`. |
| `connectx/game.py` | 330 | Stateful `Game`. The interesting part is incremental winner tracking. |

The one idea worth understanding: **`Game` tracks the winner incrementally.**
`transition` gets the landing row from `drop_row`, plays the move, then calls
`winner_at(grid, k, row, col)` — which checks only the four lines through that
one cell, `O(k)`. `terminal()` then reads a cached flag. That is why win
detection costs the same on 60x60 as on 6x7, and it is the assumption a
piece-*removing* game (Pop Out) would break.

Also in `game.py`: `state` hands out a **non-writeable view** of the grid plus a
copied `info` dict. Cheap for agents, impossible to corrupt the game through.
Check `tests/test_game.py::TestGameStateIsolation`.

### 3. Agents (~440 lines)

| File | Lines | Notes |
| --- | ---: | --- |
| `agents/types.py` | 96 | `Agent` protocol (just `select`) + `BaseAgent` (seeding, `reset`, `observe`). |
| `agents/random.py` | 16 | Read it — it shows the seeding convention. |
| `agents/greedy.py` | 59 | Win / block / centre. Works for any player count. |
| `agents/search.py` | 163 | `Position`, the engine-agnostic push/pop cursor every search uses. |
| `agents/heuristics.py` | 89 | **Scrutinize.** The weights are hand-picked. |
| `agents/minimax.py` | 176 | Negamax + alpha-beta + transposition table. Two players. |
| `agents/rollout.py` | 65 | JIT playout, the MCTS inner loop. |
| `agents/mcts.py` | 208 | UCT with max^n backups. Any player count. |

`minimax.py` is written as **negamax** rather than an explicit max/min pair
specifically because the original bug was a mismatched
`maximizing_token`/`minimizing_token` argument pair. Negamax has no such pair to
desynchronize. Worth confirming you find that convincing.

`BaseAgent.reset(config)` is where variant-specific state must be derived — not
`__init__`. An agent that takes its config at construction can only play one
variant, which makes it unusable in a sweep.

No agent touches the drop primitives any more. They build a private engine from
`self.engine_factory` and walk it through `agents.search.Position`, which
exposes only push / pop / legal / terminal / winner. That is what makes the
ladder work on a game with different dynamics — and it is verified, not assumed:
see `TestAgentsGeneralize` and `TestSolvedGameOracle`.

### 4. Data and training (~400 lines)

| File | Lines | Notes |
| --- | ---: | --- |
| `connectx/trajectory.py` | 139 | Episodes stored as **move lists**, not boards. |
| `connectx/encoding.py` | 67 | Perspective-relative planes, masks, mirror. |
| `connectx/dataset.py` | 192 | Shards + `build_supervised`. |
| `connectx/vector.py` | 174 | `VecGame`, batched stepping. |

Two design points to check:

- `Trajectory.replay(engine)` regenerates positions by **replaying moves through
  the engine**. It used to hard-code gravity, which would have silently produced
  wrong boards for a free-placement game. This is the single most important fix
  for your section-2 work.
- `encoding.encode_state` puts the **player to move in plane 0** always, and the
  plane count depends on the player count, not the board. That is what lets one
  convolutional network span board sizes. Policy width comes from
  `engine.action_space_size`, not from `cols`.

### 5. Measurement (~960 lines) — the part to review hardest

| File | Lines | Notes |
| --- | ---: | --- |
| `connectx/results.py` | 106 | Provenance and JSONL. |
| `connectx/arena.py` | 511 | Statistics, matches, tournaments, Elo. |
| `connectx/variants.py` | 350 | Variant grids, sweeps, the generalization surface. |

This is where a subtle bug would quietly corrupt months of results, so read it
with the most suspicion. Specifically:

- `_seating_for` and the seat-rotation logic in `play_match`. Connect games are
  strongly first-player biased; without rotation you measure the chair.
- `_derive_seed` — per-game seeds derive from `(match seed, game index, seat)`,
  which is why worker count cannot change a result. Confirm you believe the
  mixing function is adequate (see [soft spots](#soft-spots-and-untested-corners)).
- `elo_ratings` — Bradley-Terry by minorization-maximization, with a prior of 2
  virtual games so a perfect record still yields a finite rating.
- `games_needed` / `resolvable_gap` — the power calculation. **Check my
  arithmetic**; the whole "is this result real" story rests on it.

### 6. Interface and tooling (~700 lines)

`connectx/cli.py` (314), `connectx/benchmark.py` (193), both adapters (281),
`connectx/renderer.py` (74), `connectx/utils.py` (65). Mostly mechanical. The
adapters pass PettingZoo's `api_test` and Gymnasium's `check_env` — see
`tests/test_adapters.py`.

### 7. Tests (2,675 lines, 513 tests)

| File | Tests | Read it for |
| --- | ---: | --- |
| `tests/test_engine_conformance.py` | **186** | **The contract for a new game. Start here.** 26 of them run per engine. |
| `tests/test_agents.py` | 41 | Regression tests for the minimax bug, at every depth |
| `tests/test_game.py` | 41 | Engine semantics, state isolation, outcomes |
| `tests/test_arena.py` | 32 | Statistics, seat balance, worker determinism |
| `tests/test_results.py` | 32 | Provenance, sample size, resume |
| `tests/test_functional.py` | 29 | Primitives, incremental-vs-full-scan equivalence |
| `tests/test_cli.py` | 25 | Every command |
| `tests/test_variants.py` | 23 | Sweeps and surfaces |
| `tests/test_config.py` | 18 | Every rejected config |
| `tests/test_encoding.py` | 17 | Planes, perspective, mirror algebra |
| `tests/test_trajectory.py` | 16 | Replay, returns |
| `tests/test_vector.py` | 15 | Batched semantics vs scalar |
| `tests/test_adapters.py` | 14 | Upstream conformance suites |
| `tests/test_dataset.py` | 13 | Shard roundtrip, supervised targets |
| `tests/test_utils.py` | 8 | Serialization |

If you read one test file, make it `test_engine_conformance.py` — it is the
executable specification of what a game has to do, and it is how you will verify
a Gomoku engine. Adding `("MyGame", MyGame, config)` to `ENGINES` runs all 26
per-engine checks against it.

It also runs the **whole agent ladder** against every engine, so a new game
gets working baselines rather than only a verified contract. `tests/free_placement.py`
is there to keep that claim honest: an engine with no gravity and a
`rows * cols` action space, which anything assuming "action index == column"
fails on.

---

## The bugs that were fixed

Each of these was reproduced before the fix and has a regression test. If you
want to confirm a fix is real, check out the parent commit and run the repro.

| # | Bug | Evidence it was real | Test |
| --- | --- | --- | --- |
| 1 | A winning move that also filled the board was reported as a **tie** | `[[1,1,1,1]]` → `{'winner': None, 'tie': True}` | `test_game.py::test_win_that_also_fills_the_board_is_not_a_tie` |
| 2 | Minimax scored **the opponent's pieces as its own** at odd depths | `evaluate` got `(max, min) = (1,1)` at depths 1, 3, 5 — including the default depth 3 | `test_agents.py::test_takes_immediate_win_at_every_depth` |
| 3 | Board dims and `k` **wrapped at 255** | `create_grid((300, 7))` → `(44, 7)` | `test_functional.py::test_dimensions_above_255_are_not_truncated` |
| 4 | **No config validation** | `k=9` on 6x7 (unwinnable), token `0` (invisible), duplicate tokens, 1 player, zero-size boards all accepted | `test_config.py` (9 rejected configs) |
| 5 | `place_token` **silently dropped moves** on a full column | Functional loop advanced the clock on a move that never happened | `test_functional.py::TestDropRowAndLegality` |
| 6 | `Game.state` handed out **live internal objects** | `state["info"]["time"] = 999` changed the game clock | `test_game.py::TestGameStateIsolation` |
| 7 | `RandomAgent` used the **global numpy RNG** | Nothing reproducible; parallel workers shared state | `test_agents.py::test_does_not_touch_the_global_rng` |
| 8 | `terminal()` **rescanned the whole board** every call | 330 µs at 60x60 | `connectx bench` |
| 9 | **No alpha-beta** | depth 5 cost 6.0 s per move | `test_agents.py::test_alpha_beta_prunes` |
| 10 | `Trajectory.replay` **assumed gravity** | Would silently mislabel training data for any other game | `test_engine_conformance.py::TestRecording` |
| 11 | `GameEngine` was **decorative** | Nothing consumed it; every caller built `Game` directly | `test_engine_conformance.py` + `engine=` everywhere |
| 12 | Results **recorded no seed** | A record could not reproduce itself | `test_results.py::TestResultProvenance` |
| 14 | `Game.transition` **discarded the seat a resumed state was started with** | It recomputed `active = time % n_players`, so any position where the two disagreed silently corrected itself on the next move | `test_game.py::TestGameStart::test_resumed_seat_is_honoured` |
| 13 | `isinstance(x, GameEngine)` **raised** on Python <= 3.11 | CI red on 3.10/3.11, green on 3.12+; `hasattr` evaluates the `state` property, which raises before `start()` | `test_engine_conformance.py::TestProtocol::test_protocol_check_survives_an_unstarted_engine` |

Measured effects of 8 and 9:

| | before | after |
| --- | ---: | ---: |
| `terminal()` at 60x60 | 330 µs | 0.15 µs |
| minimax depth 5, one move | 6.0 s | 0.002 s |
| MCTS, per simulation | 19.9 µs | 9.8 µs |

---

## Decisions you may want to overturn

These are judgement calls, not facts. Each is cheap to reverse now and expensive
later.

1. **`Action` is a plain `int` (a column index), not `np.uint8`.** Simpler, but
   it means the type gives you no protection against passing a row.
2. **`state` returns a read-only *view*, not a copy.** Cheap and safe, but a
   caller who wants to mutate must `np.array(grid)` and may be surprised.
3. **Rewards are terminal-only**, via `RewardSpec(win, loss, draw)`. No shaping,
   no per-step penalty. If you want reward shaping it goes in a wrapper, not the
   engine — but you may prefer it in the engine.
4. **Episodes are stored as move lists**, not boards. Tiny files, but every
   read costs a replay. If you end up I/O-bound rather than storage-bound,
   invert this.
5. **Elo uses a 2-game prior.** It keeps perfect records finite, but it also
   shrinks extreme ratings toward the field. Change `prior_games` if you would
   rather see the raw fit.
6. **Seat rotation is cyclic**, so with `n > 2` players it balances seat
   occupancy but samples only `n` of the `n!` orderings. Fine for first-player
   bias, insufficient if relative ordering matters.
7. **`play_match` refuses game counts that are not a multiple of the seat
   count** when rotating. Strict, and it will annoy you; the alternative is a
   silently unbalanced match.
8. **`transition` raises after the game ends** rather than being a no-op. Louder
   than most RL environments.
9. **numba is a hard dependency.** Justified by measurement
   (`scripts/ablation.py`, 14x on a real match), but it costs ~155 MB and pins
   the usable numpy range. The whole suite passes with `NUMBA_DISABLE_JIT=1`, so
   dropping it is a real option if it ever blocks a Python upgrade.

---

## Soft spots and untested corners

Where I would look first for something wrong.

**1. The heuristic weights are unfitted.** `agents/heuristics.py` contains
`_THREAT_BONUS = 40.0`, `_DEFENSE = 1.15`, and a `2.0` multiplier on the centre
term. I chose those by reasoning, not by tuning, and never validated them across
variants. Minimax's playing strength — and therefore every Elo number involving
it — depends on them. This is the least-defensible code in the project.

**2. MCTS `exploration = 1.4` is also unfitted**, and the right value depends on
the reward scale, which changes if you change `RewardSpec`.

**3. Ground truth for playing strength is thin.** Agents are mostly measured
against each other, which cannot tell you whether any of them play *well*. There
is now one absolute check — `TestSolvedGameOracle` searches tic-tac-toe out at
depth 9 and confirms perfect play draws every game and never loses to anything —
but nothing comparable exists for the drop games, where the trees are too large
to solve. A solved-position oracle for small drop boards is still in `todo.md`.

**4. `_derive_seed` is an ad-hoc mixing function**, not a real hash:
`(base * 1_000_003 + game_index * 9_176 + seat * 31) % (2**31 - 1)`. It is
deterministic and I saw no collisions in practice, but if you care, replace it
with something principled before generating data you intend to keep.

**5. Genuinely thin test coverage** (with the JIT disabled, so these are real):

| Module | Coverage | Why |
| --- | ---: | --- |
| `agents/human.py` | 25% | Reads stdin; only the error paths are hard to reach |
| `connectx/renderer.py` | 67% | Uncovered: the >8-player symbol fallback, and **the winning-line highlight geometry**, which is a real code path nothing exercises |

Everything else is 94–100%.

**6. Elo across variants is not comparable.** Each variant is fitted
independently and anchored to the same mean. The tables say so, but it is an
easy mistake to make when reading a surface.

**7. Search costs ~2.7x more than it did.** Routing the agents through the
engine made them general and slower: `minimax:depth=4` went from 0.32 ms to
0.86 ms per move, `greedy` from 0.023 ms to 0.084 ms. Node counts are identical,
so the search is unchanged — the cost is `Game.transition` being Python-bound
(about 1.4 us of compiled work inside a 4.1 us call). Judged worth it, since
these are baselines rather than the research target, but reverse the judgement
if it ever bounds an experiment.

**8. Use `implements_engine`, not `isinstance`, against `GameEngine`.** On
Python 3.11 and earlier, `isinstance` against a runtime-checkable protocol calls
`hasattr` on every member, which evaluates properties — and `Game.state` raises
before `start()`. `implements_engine` inspects the type statically and behaves
identically on every version. This is the one cross-version landmine found so
far, and it was found by CI rather than by me.

**9. The PettingZoo adapter emits two upstream warnings** about `Dict`
observation spaces. Benign — PettingZoo's own classic environments trigger the
same — but they will show up in your logs.

**10. `VecGame` has no agent interface.** It is a raw batched stepper; you drive
it with your own policy. Intentional, but it means the agent ladder does not run
inside it.

---

## Claims made in the docs, and their evidence

Every performance number in the README is reproducible. If one does not
reproduce on your machine, the claim is wrong, not your setup.

| Claim | Reproduce with |
| --- | --- |
| Win detection is O(1) after an O(k) update | `connectx bench` |
| `VecGame` reaches 6.5M moves/s | `connectx bench` |
| numba is worth 14x on a real match | `python scripts/ablation.py` and again with `NUMBA_DISABLE_JIT=1` |
| Adapters conform to their upstream APIs | `pytest tests/test_adapters.py` |
| Worker count never changes a result | `test_arena.py::test_deterministic_across_worker_counts` |
| `k=3` variants are forced first-player wins | `connectx match --shape 5 6 --k 3 --agents minimax:depth=4 mcts:simulations=400 --games 200 --workers 8` → seat wins 200–0 |

**One correction you should know about.** An earlier version of the README
claimed the agent ladder *reordered* across variants — that MCTS beat minimax on
small boards. That was wrong. It came from a 40-game-per-pairing run, which can
only resolve ~160 Elo, and the claimed gap was 42. Re-running at 600 games gives
exactly 0. The `games_needed` helper was written afterwards and immediately
flagged it. The current README states the corrected finding, and `CHANGELOG.md`
records the retraction. I mention it because it is the clearest evidence of what
this harness is for — and because you should assume other unverified claims of
mine could be wrong in the same way.

---

## Commit-by-commit history

Twenty-six commits, each independently green (verified by staging, stashing the
rest, and running the suite). `git log --oneline 6ac378f..HEAD`.

| Commit | What it did |
| --- | --- |
| `2b9958d` | Fixed the `place_token` overflow and the pre-existing minimax errors |
| `ee2daee` | Engine protocol, trajectories, replay memory |
| `7c3b9d8` | Game recording, undo, forking |
| `53fc445` | **The correctness rework** — bugs 1–7, config validation, agent interface |
| `73b94d1` | Training layer: encoding, `VecGame`, datasets, adapters |
| `0f6fbd9` | Measurement harness: matches, Elo, intervals, sweeps |
| `c32f8dc` | CLI and experiment runner |
| `981ea17` | Packaging, lint, types, CI |
| `4af1fef` | USAGE.md, README, changelog |
| `cac61da` | MCTS 2x, move-for-move identical |
| `62367ea` | The numba ablation |
| `fadc401` | Generalization surface |
| `1a44bfc` | Documented the surface result |
| `fbc16a4` | **Retracted** the false `GameEngine` claim in the docs |
| `d685872` | Provenance, sample-size planning, resumable runs |
| `dbbfbd5` | Made the `GameEngine` protocol load-bearing |
| `9ddd9b3` | **Retracted** the false headline result |

The two retractions are the commits worth reading, because they show what was
overclaimed and how it was caught.

---

## Review checklist

A suggested order for your own pass. Roughly a day if you read carefully.

- [ ] `pip install -e ".[dev]"`, then `pytest` — 513 tests should pass in ~2 s.
- [ ] `NUMBA_DISABLE_JIT=1 pytest --cov --cov-report=term-missing` — the honest
      coverage picture (95%).
- [ ] Read `connectx/types.py` and `connectx/engine.py`. Confirm the seat/token
      distinction is clear to you; it is the main source of subtle error.
- [ ] Read `connectx/game.py::transition` and convince yourself the incremental
      winner tracking is correct. Compare against
      `test_functional.py::test_winner_at_matches_full_scan`, which fuzzes 200
      random games checking incremental against full scan.
- [ ] Read `connectx/arena.py` end to end. This is where a quiet bug does the
      most damage. Check `_seating_for`, `_derive_seed`, and `games_needed`.
- [ ] Read `tests/test_engine_conformance.py`. This is the contract your Gomoku
      engine will have to satisfy — decide now whether it asks the right things.
- [ ] Decide on the heuristic weights in `agents/heuristics.py`: tune them, or
      accept them and note the caveat in anything you publish.
- [ ] Run `connectx bench` and `python scripts/ablation.py` and check the
      numbers match the README on your hardware.
- [ ] Skim `CHANGELOG.md` for the retractions.
- [ ] Check GitHub Actions is green. The first run was red on Python 3.10 and
      3.11 only; see bug 13. The matrix now covers 3.10-3.14 and runs the suite
      twice per version, with and without the JIT.
- [ ] Read `todo.md` sections 2 and 3, and decide the PyTorch/JAX question.

When you find something wrong — and on 2,300 lines of new code you will —
`tests/` is the right place to start the fix, because almost every claim in this
document is pinned by a test that would go red.

---

## Keeping this guide honest

A review guide that lies about the codebase is worse than no guide, so the
parts of it that can rot are machine-checked.

```bash
python scripts/check_guide.py         # report drift, exit 1 if any
python scripts/check_guide.py --fix   # rewrite the numbers from reality
```

It re-derives the test counts, the conformance counts, the test line total, and
every module size quoted in the reading-order tables, then diffs them against
what the docs claim. It runs in CI, so a pull request that adds a module or a
test and forgets the guide goes red.

**What is volatile and gets checked automatically**

| Fact | Where | Regenerate with |
| --- | --- | --- |
| Total test count | this file, `USAGE.md` | `--fix` |
| Conformance counts | this file, `USAGE.md` | `--fix` |
| Test line total | this file | `--fix` |
| Module sizes | reading-order tables | `--fix` |

**What is volatile and is *not* checked** — update these by hand:

| Fact | Refresh with |
| --- | --- |
| Coverage percentages | `NUMBA_DISABLE_JIT=1 pytest --cov` |
| Performance numbers | `connectx bench`, `python scripts/ablation.py` |
| The commit table | `git log --oneline 6ac378f..HEAD` |
| Per-file test counts | `for f in tests/test_*.py; do pytest "$f"; done` |

Prose is not checkable and is not attempted. The sections most likely to go
stale as *judgement* rather than as fact are
[decisions you may want to overturn](#decisions-you-may-want-to-overturn) and
[soft spots](#soft-spots-and-untested-corners) — when you fix a soft spot or
reverse a decision, delete the entry rather than leaving it to mislead.

**When you extend the project**, the entries worth adding are the ones this
guide is built around: a bug you fixed and how you reproduced it, a decision you
made that a future reader might reasonably disagree with, and anything you know
is weak. Those are the parts that were useful to write and are impossible to
recover later.
