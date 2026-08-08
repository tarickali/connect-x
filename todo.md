# Todo List

## Next up

### 1. Neural agent and an AlphaZero-style loop

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

### 2. The generalization matrix

The headline result. Needs (1) first.

- [ ] Train on variant A, evaluate on the whole grid, plot the transfer heatmap.
- [ ] Compare against the algorithmic baseline surface already produced by
      `connectx surface`, which is the "no learning" control.
- [ ] Report transfer along each axis separately — board size, `k`, and player
      count are different kinds of distribution shift and probably behave
      differently.

### 3. Measurement improvements

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

### 4. Engine and agents

- [ ] Zobrist hashing to replace `grid.tobytes()` transposition-table keys.
- [ ] `max^n` or paranoid search, so multiplayer variants get a strong
      non-Monte-Carlo baseline.
- [ ] Opening book.
- [ ] Optional PyGame renderer (`connectx.renderer` is the seam).
- [ ] Variants with different transition rules — Pop Out, Pop 10, Power Up — as
      separate `GameEngine` implementations.
- [ ] Other m,n,k games: Gomoku, Connect6, Pente.

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
