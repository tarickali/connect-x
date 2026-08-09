# Todo List

## 0. Before serious research runs start — done

- [x] **Record provenance in every result.** Today a JSONL record holds the
      config, the specs, and the numbers — but **not the seed**, so a result
      cannot be reproduced from its own record. Add seed, UTC timestamp, git
      SHA, `connectx.__version__`, and the numpy/numba versions to
      `MatchResult.to_dict`, `TournamentResult.to_dict`, and the surface
      records. This is the single highest-value item on this page: in three
      months there will be forty result files and no way to tell which code
      produced them.
- [x] **Sample-size helper.** `games_needed` / `resolvable_gap`. Immediately
      caught a false claim in the README's own headline result.
- [x] **Resumable long runs.** `sweep(jsonl=..., resume=True)` and `--resume`.

## 1. Finish the GameEngine seam — done

- [x] Every consumer takes an `engine=` factory: `play_game`, `play_match`,
      `round_robin`, `sweep`, `tournament_surface`, both adapters, benchmarks.
- [x] `Trajectory.replay(engine)` replays through the engine instead of
      assuming gravity; `build_supervised` threads it through.
- [x] `GameEngine.action_space_size` drives policy width, so a wider action
      space no longer needs a retrofit. `mirror_action` is documented as
      column-only and `build_supervised` rejects mirroring when the action space
      is not per-column.
- [x] Confirmed `validate_config` passes unknown keys through, so a new game can
      carry extra config fields. Covered by a test.
- [x] `tests/test_engine_conformance.py`: 73 dynamics-agnostic checks covering
      the protocol, state isolation, outcomes, recording, and integration with
      the arena and dataset pipeline. **Add a new game to `ENGINES` and it is
      checked automatically.**

Remaining for a *free-placement* game specifically (do with section 2):

- [ ] 2D mirror for the free-placement action space.
- [ ] `Config` fields for the new dynamics (`gravity`, exact-`k`, stones/turn).

## 2. Rule diversity: the m,n,k family

Today every variant is *one* game with three scalar knobs — shape, `k`, and
player count. That is real diversity for **scale** generalization, and the
surface result shows it produces findings. It is not diversity of **dynamics**:
every variant shares gravity, a column-sized action space, no removal, and one
stone per turn. Ordered by cost.

- [ ] **Misère** (making `k` in a row *loses*). One flag flipping the reward
      sign; no engine change at all. Genuinely different optimal play, so it is
      by far the cheapest real rule-generalization axis available.
- [ ] **Obstacles / blocked cells.** Pre-fill cells with a reserved token that
      belongs to no player. Cheap, and changes board geometry without touching
      dynamics.
- [ ] **Free placement** (`gravity=False`) — the big unlock. Action space
      becomes `rows * cols`. Gets tic-tac-toe, **Gomoku** (15x15, k=5), and the
      general m,n,k game. Needs section 1 done first: Gomoku is *not*
      expressible today, because `Game` drops tokens down columns.
- [ ] **Gomoku rule details** once free placement exists: exact-`k` versus
      overlines (whether six in a row counts as a win), and optionally an
      opening rule such as swap2, since plain 15x15 Gomoku is a first-player
      win and heavily seat-biased.
- [ ] **Connect6** — free placement, two stones per turn after the first move.
      Introduces variable moves-per-turn, which the `StepResult` shape and the
      PettingZoo adapter both currently assume away.
- [ ] **Pop Out / Pop 10** — gravity plus removal from the bottom. Action space
      roughly doubles (drop *or* pop) and the grid mutates below existing
      pieces, which breaks the "tokens never move once placed" assumption that
      incremental win detection relies on.
- [ ] **Pente** — free placement plus custodian capture. Removal again, plus a
      second win condition (capture count), so `Report` needs to carry more than
      a line.
- [ ] **Order and Chaos / Wild** — either player may place either token. Action
      space doubles and the two seats have different objectives, which breaks
      the zero-sum assumption baked into `RewardSpec` defaults and negamax.

## 3. Next up

### 3.1 Neural agent and an AlphaZero-style loop

The gate on everything this project exists to measure. All the scaffolding is
already in place — `connectx.encoding` emits perspective-relative planes whose
count depends on the player count rather than the board, `VecGame` supplies
self-play throughput, and `build_supervised` produces policy and value targets
with mirror augmentation.

- [ ] Decide **PyTorch or JAX**, and keep it behind an `[nn]` extra so the
      engine stays dependency-light.
- [ ] Fully-convolutional policy/value net, so one set of weights spans board
      sizes. No dense layer over a flattened board — that would pin the model
      to a single shape and defeat the point.
- [ ] Self-play loop: `VecGame` → `ReplayMemory` → `build_supervised` → fit →
      new checkpoint into the agent registry.
- [ ] Register checkpoints as agent specs (`neural:ckpt=runs/a/best.pt`) so
      they drop straight into `arena`, `surface`, and the Gymnasium adapter's
      `opponent=` without special-casing.
- [ ] MCTS guided by the network (replace random rollouts with the value head).

### 3.2 The generalization matrix

The headline result. Needs 3.1 first.

- [ ] Train on variant A, evaluate on the whole grid, plot the transfer heatmap.
- [ ] Compare against the algorithmic baseline surface already produced by
      `connectx surface`, which is the "no learning" control.
- [ ] Report transfer along each axis separately — board size, `k`, and player
      count are different kinds of distribution shift and probably behave
      differently.

### 3.3 Measurement improvements

- [ ] Equal-*time* rather than equal-depth matchups as the default comparison.
      `MinimaxAgent(time_limit=...)` and `MCTSAgent(time_limit=...)` already
      support it; the arena should expose a budget rather than a depth.
- [ ] Full seat permutations for variants with more than two players. The
      current cyclic rotation balances seat occupancy but only samples `n` of
      the `n!` orderings.
- [ ] Track benchmark numbers across commits to catch performance regressions.
- [ ] Plotting helpers for the sweep and surface JSONL: heatmaps, Elo tables,
      throughput curves. Behind a `[plot]` extra so matplotlib stays optional.
- [ ] A solved-position oracle for small variants, to measure *absolute* rather
      than relative strength.

### 3.4 Engine and agents

- [ ] Zobrist hashing to replace `grid.tobytes()` transposition-table keys.
- [ ] `max^n` or paranoid search, so multiplayer variants get a strong
      non-Monte-Carlo baseline.
- [ ] Opening book.
- [ ] Optional PyGame renderer (`connectx.renderer` is the seam).

## Deliberately deferred

- **Bitboards.** The numba ablation showed the scalar `Game` path is
  Python-bound, not compute-bound: its four compiled calls sum to ~1.8 µs of a
  4.1 µs transition, so the remaining 55% is dict and view construction.
  Bitboards would optimize the layer that is already fast. `VecGame` already
  gives 20x by amortizing the Python overhead across a batch; reach for that
  instead. Revisit only if profiling shows the compiled primitives dominating.
- **More game variants before a learning agent exists.** The `GameEngine` seam
  makes them cheap to add later. Adding them now widens the benchmark without
  deepening the result.

## Done

- [x] Add benchmark suite
- [x] Fully tested!
- [x] Recording games (`Trajectory`, `ReplayMemory`, `connectx.dataset`)
- [x] Allow parallel environments (`VecGame`, plus process-parallel matches)
- [x] Config validation
- [x] Incremental win detection
- [x] Alpha-beta minimax, MCTS, greedy baselines
- [x] Rewards, seeding, observation encoding
- [x] PettingZoo and Gymnasium adapters
- [x] Measurement harness: matches, Elo, confidence intervals, sweeps
- [x] CLI and config-driven experiments
- [x] CI, lint, type checking, USAGE.md
- [x] Justify the numba dependency with measurements (`scripts/ablation.py`)
- [x] Optimize the MCTS tree (2x; the rollout was never the bottleneck)
- [x] Generalization surface: round robin on every variant (`connectx surface`)
