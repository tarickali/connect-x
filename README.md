# connect-x

[![CI](https://github.com/tarickali/connect-x/actions/workflows/ci.yml/badge.svg)](https://github.com/tarickali/connect-x/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%E2%80%93%203.13-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)

A **parameterized family of Connect-style games** for training and evaluating AI
agents, built to measure how well an agent generalizes when the task changes
underneath it.

Board shape, win length `k`, and player count are all configuration. That turns
a single game into a universe of related ones — and makes the interesting
question askable: *does an agent that is strong on 6x7 stay strong on 9x10, or
with five in a row, or with three players?*

```bash
pip install -e ".[dev]"
connectx tournament --preset connect4 --agents random greedy mcts minimax:depth=4
connectx sweep --agents minimax:depth=4 greedy --shapes 5x6 6x7 9x10 --ks 3 4 5
```

---

## Why this exists

Agent benchmarks tend to measure competence on one fixed task. Generalization
research needs the opposite: a family of tasks that share structure but differ
in ways you control. Connect 4 is a good base — simple enough to reason about,
deep enough to be non-trivial — and it sits inside the well-studied family of
*m,n,k-games*, so widening it is principled rather than arbitrary.

The measurement machinery is treated as part of the library, not as scripts
bolted on afterwards: seat rotation, seeded reproducibility, confidence
intervals, and variant sweeps are first-class.

---

## What generalization looks like here

A full round robin on twelve variants — 12 boards × 6 pairings × 40 games,
2,880 games, seats rotated throughout. **The ladder reorders.**

| agent | 5x6 k=3 | 5x6 k=4 | 6x7 k=4 | 9x10 k=5 | 12x12 k=5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `minimax:depth=4` | 1737 | 1798 | **2033** | **2185** | **2209** |
| `mcts:simulations=400` | **1779** | **1827** | 1883 | 1779 | 1809 |
| `greedy` | 1623 | 1446 | 1369 | 1345 | 1316 |
| `random` | 862 | 929 | 716 | 690 | 665 |
| *minimax − mcts* | *−42* | *−29* | *+150* | *+406* | *+400* |

MCTS is ahead on the small boards and 400 Elo behind on the large ones. Nothing
about either agent changed — only the board did. Pick 5x6 as your benchmark and
you conclude MCTS is stronger; pick 12x12 and you conclude the opposite.

The effect is not a compute artifact, which makes it sharper: `mcts:simulations=400`
takes ~3.6 ms per move against minimax's ~0.32 ms, so the agent that loses on
large boards is the one spending **11× more time**. A fixed simulation budget
covers a vanishing fraction of a growing tree, while fixed-depth tactical search
stays exact.

The `spread` row of the full table carries a second result: the Elo range
between best and worst agent runs from 757 (5x6 k=5) to 1544 (12x12 k=5). Some
variants barely separate these agents at all — worth knowing before you use one
as a benchmark.

```bash
connectx surface --agents random greedy minimax:depth=4 mcts:simulations=400 \
  --shapes 5x6 6x7 9x10 12x12 --ks 3 4 5 --games 40 --workers 8
```

Two caveats the harness makes explicit rather than hiding: ratings are fitted
per variant and anchored to the same mean, so **only differences within a column
mean anything**; and this is an equal-*budget* comparison, not equal-time. An
equal-time mode is on the roadmap.

---

## Features

- **Variants as configuration** — board shape, win length `k`, and player count,
  validated so an unwinnable game is rejected rather than silently played.
- **Two APIs** — pure JIT-compiled functions for custom training loops, and a
  stateful `Game` for scripting. Both back onto the same primitives.
- **Fast** — incremental win detection makes `terminal()` O(1) after an O(k)
  update; batched `VecGame` reaches **6.5M moves/second**.
- **RL-ready** — per-seat rewards, seeded determinism, perspective-relative
  observation planes, action masks, mirror augmentation, trajectory recording.
- **Standard APIs** — PettingZoo (AEC) and Gymnasium adapters, both passing
  their upstream conformance tests.
- **An agent ladder** — random → greedy → alpha-beta minimax → UCT MCTS, so a
  win rate means something.
- **A measurement harness** — matches with Wilson intervals, round-robin
  tournaments with Elo, variant sweeps, generalization surfaces, JSONL results,
  process parallelism.

---

## Install

Python 3.10+.

```bash
git clone https://github.com/tarickali/connect-x.git
cd connect-x
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"        # or ".[rl]" for just the RL adapters
```

Full command reference: **[USAGE.md](USAGE.md)**.

---

## Quick start

```python
from connectx import Game, preset
from agents import make_agent

config = preset("connect4")                       # or make_config((9, 10), 5, [1, 2, 3])
agents = [make_agent("minimax:depth=6", config, seed=0),
          make_agent("mcts:simulations=800", config, seed=1)]

game = Game(config)
state, actions = game.start()
while not game.terminal():
    action = agents[state["info"]["active"]].select(state, actions)
    result = game.step(action)                    # rewards + terminated included
    state, actions = result.state, result.actions

game.render()
print(game.report())      # {'winner': {'token': 1, 'id': 0}, 'steps': 31, 'tie': False}
print(game.rewards())     # array([ 1., -1.])
```

Train against the standard APIs:

```python
from connectx.adapters import make_aec_env, make_gym_env

env = make_aec_env(config)                                    # PettingZoo, self-play
env = make_gym_env(config, opponent="minimax:depth=4")        # Gymnasium, single agent
```

---

## Architecture

```
config  ──►  functional  ──►  Game  ──►  VecGame        (engine)
   │         (numba)          │
   │                          ├──►  Trajectory ──► dataset      (data)
   │                          ├──►  encoding             (observations)
   │                          └──►  adapters        (PettingZoo / Gymnasium)
   │
   └──►  variants ──► arena ──► Elo, Wilson intervals, JSONL    (measurement)

agents: Agent protocol ──► random │ greedy │ minimax │ mcts │ human
```

| Layer | Module | Responsibility |
| --- | --- | --- |
| Config | `connectx.config` | Variants, validation, presets |
| Core | `connectx.functional` | Pure JIT primitives; no hidden state |
| Engine | `connectx.game` | Stateful `Game`; undo, fork, record, rewards |
| Protocol | `connectx.engine` | `GameEngine` — implement it to add a new game |
| Batch | `connectx.vector` | `VecGame`, N boards in one kernel |
| Data | `connectx.trajectory`, `connectx.dataset` | Episodes, replay, dataset shards |
| Learning | `connectx.encoding` | Planes, masks, mirror symmetry |
| Bridges | `connectx.adapters` | PettingZoo AEC, Gymnasium |
| Measurement | `connectx.arena`, `connectx.variants` | Matches, tournaments, sweeps |
| Agents | `agents` | Baseline ladder + registry |

The engine never imports agent code, and agents never import the harness — so a
new game, a new agent, and a new experiment are three independent changes.

---

## Performance

Apple M-series, single core, numba warm. Reproduce with `connectx bench`.

| Operation | 6x7 | 20x20 | 60x60 |
| --- | --- | --- | --- |
| `winner_at` (incremental) | 0.15 µs | 0.15 µs | 0.15 µs |
| `winner` (full scan) | 0.22 µs | 0.35 µs | 1.98 µs |
| `Game.transition` | 4.3 µs | 4.1 µs | 4.7 µs |

Win detection is incremental — only the four lines through the cell just filled
are examined — so `terminal()` costs the same on a 60x60 board as on a 6x7 one.

| Environment | Moves/second |
| --- | --- |
| `Game` (scalar) | 232,000 |
| `VecGame`, 256 envs | 3,748,000 |
| `VecGame`, 2048 envs | 6,487,000 |

| Agent | ms per move (cold) |
| --- | --- |
| `random` | 0.002 |
| `greedy` | 0.023 |
| `minimax:depth=4` | 0.313 |
| `mcts:simulations=200` | 1.814 |

### Why numba

numba is the only non-trivial dependency, so it was measured rather than
assumed. `scripts/ablation.py` compares the shipped code against the same source
with `NUMBA_DISABLE_JIT=1` *and* against an idiomatic vectorized numpy
implementation — the second comparison is the honest one.

| workload | numba | alternative | speedup |
| --- | ---: | ---: | ---: |
| 200-game match | 1.8 s | 25.1 s | **14×** |
| `minimax:depth=4`, one move | 0.31 ms | 6.39 ms | **20×** |
| `VecGame`, 1024 envs | 4.87M moves/s | 0.65M (numpy) | **7.5×** |
| `winner` full scan, 6x7 | 0.28 µs | 40.4 µs (numpy) | **145×** |
| `generate_actions`, 60x60 | 0.83 µs | 0.88 µs (numpy) | 1.07× |

The pattern is consistent: numba wins where numpy cannot vectorize — scalar,
branchy, early-exit loops. A full-board scan stops at the first line it finds;
numpy has to materialize every window before testing any. Where the work *is* a
clean vectorized expression, it is a tie.

Cost: ~155 MB of wheels, +0.16 s on a warm import, ~2 s cold, and coupling to
numba's supported numpy range. Worth it at 14× on the workload this project
actually runs. **All 275 tests pass under `NUMBA_DISABLE_JIT=1`** — numba is a
pure accelerator here, never load-bearing for behaviour, so the ablation is one
environment variable and there is an escape hatch if it ever blocks an upgrade.

---

## Configuration

| Option | Type | Rule |
| --- | --- | --- |
| `shape` | `(rows, cols)` | Both positive; no upper bound |
| `k` | `int` | `1 <= k <= max(rows, cols)`, so a win can fit |
| `players` | `list[int]` | ≥ 2 distinct tokens in `1..255` (`0` marks empty) |

Classic Connect 4 is `shape=(6, 7)`, `k=4`, `players=[1, 2]`. Presets: `tiny`,
`small`, `connect4`, `connect5`, `wide`, `tall`, `three-player`, `four-player`.

---

## Extending

`connectx.engine.GameEngine` is the seam. Implement it and the arena, adapters,
recorders, and dataset tooling all work unchanged — which is how the wider
*m,n,k* family (Gomoku, Connect6, Pente) and the Connect 4 rule variants (Pop
Out, Pop 10, Power Up) are meant to be added.

Adding an agent is smaller: subclass `BaseAgent`, implement `select`, register
it. Derive variant-specific state in `reset(config)` rather than `__init__` so a
single instance can play any variant — otherwise it cannot appear in a sweep.

See [USAGE.md](USAGE.md#extending-the-project).

---

## Contributing

Contributions are welcome — open an issue or a pull request. CI runs tests,
lint, type checks, and a smoke test of every documented command.

## License

Apache License 2.0. See [LICENSE](LICENSE).
