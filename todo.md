# Todo List

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

## Next

### Engine

- [ ] Optional PyGame renderer (ASCII is done; `connectx.renderer` is the seam).
- [ ] Zobrist hashing, to replace `grid.tobytes()` keys in the transposition table.
- [ ] Bitboard representation for the two-player case — likely another 5-10x on
      `VecGame`, at the cost of a second code path.
- [ ] Variants with different transition rules (Pop Out, Pop 10, Power Up) as
      separate `GameEngine` implementations.
- [ ] Other m,n,k games: Gomoku, Connect6, Pente.

### Agents

- [ ] Neural agent: a fully-convolutional policy/value net, so one set of
      weights spans board sizes. `connectx.encoding` already produces the right
      shape for this.
- [ ] AlphaZero-style training loop using `VecGame` + `build_supervised`.
- [ ] `max^n` or paranoid search, so multiplayer variants get a strong
      non-Monte-Carlo baseline.
- [ ] Opening book, and a solved-position oracle for small variants to measure
      absolute (not just relative) strength.

### Measurement

- [ ] Generalization matrix: train on variant A, evaluate on the full grid, and
      plot the transfer heatmap. This is the headline figure the project is for.
- [ ] Equal-time rather than equal-depth matchups as the default comparison
      (`MinimaxAgent(time_limit=...)` already supports it).
- [ ] Track benchmark numbers across commits, to catch performance regressions.
- [ ] Plotting helpers for sweep JSONL: heatmaps, Elo tables, throughput curves.
