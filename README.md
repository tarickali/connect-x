# connect-x

[![CI](https://github.com/tarickali/connect-x/actions/workflows/ci.yml/badge.svg)](https://github.com/tarickali/connect-x/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%E2%80%93%203.14-blue)](https://www.python.org/)
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

**Not every variant in a parameterized family measures anything.** That turns
out to be the first real result, and the harness is built to surface it rather
than average over it.

`minimax:depth=4` versus `mcts:simulations=400`, seats rotated, 600 games per
variant — enough to resolve a 40 Elo difference:

| variant | minimax score | 95% CI | Elo gap | seat wins | what it measures |
| --- | ---: | ---: | ---: | ---: | --- |
| 5x6 k=3 | 50.00% | [46.0%, 54.0%] | 0 ± 28 | **200–0** | nothing |
| 6x7 k=3 | 50.00% | [46.0%, 54.0%] | 0 ± 28 | **200–0** | nothing |
| 5x6 k=4 | 54.75% | [50.8%, 58.7%] | +33 | 131–35 | a little |
| 12x12 k=5 | 96.50% | [93.0%, 98.3%] | **+576** | 101–99 | a lot |

On the `k=3` boards the first player wins **every single game** — it is a forced
win that both agents find and neither loses. The entrants finish exactly 50/50
because seats were rotated, and the score is a property of the rotation, not of
either agent. Without seat rotation this variant reads as a coin flip; without
`seat_wins` reported alongside, you would never learn why.

At the other end, 12x12 k=5 has no first-player advantage left (101–99) *and*
separates the agents decisively (+576 Elo). That combination — balanced seats,
wide skill gap — is what a variant has to have to be worth benchmarking on.

Two methodological notes that fell out of building this, both now enforced in
code:

- **The 40-game version of this table said the opposite.** A coarser sweep put
  MCTS 42 Elo ahead on 5x6 k=3, and it was noise: 40 games can only resolve
  ~160 Elo. `connectx.arena.games_needed(40)` says 596 games are needed to call
  a 40 Elo difference, so the harness will now tell you before you spend the
  compute.
- The comparison is equal-*budget*, not equal-time. `mcts:simulations=400` takes
  ~3.6 ms per move against minimax's ~0.32 ms, so on large boards the agent
  losing by 576 Elo is the one spending **11× more time per move**.

```bash
connectx surface --agents random greedy minimax:depth=4 mcts:simulations=400 \
  --shapes 5x6 6x7 9x10 12x12 --ks 3 4 5 --games 600 --workers 8 \
  --jsonl results/surface.jsonl --resume
```

Ratings are fitted per variant and anchored to the same mean, so **only
differences within a column mean anything**. Every record carries its seed, git
SHA, and library versions, so any number here can be traced back to the code
that produced it.

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
  win rate means something. Every agent plays through the engine interface, so
  the ladder works on any game satisfying `GameEngine`, not just drop games.
- **A measurement harness** — matches with Wilson intervals, round-robin
  tournaments with Elo, variant sweeps, generalization surfaces, sample-size
  planning, process parallelism, and resumable runs.
- **Reproducible by construction** — every result record carries its seed, git
  SHA, and library versions; worker count never changes a number.

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
| Measurement | `connectx.arena`, `connectx.variants` | Matches, tournaments, sweeps, surfaces |
| Provenance | `connectx.results` | Seeds, versions, JSONL, resume |
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
actually runs. **The whole suite passes under `NUMBA_DISABLE_JIT=1`** — numba is a
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

`connectx.engine.GameEngine` describes the interface a game must present, and
`Game` implements it. It is the intended seam for the wider *m,n,k* family
(Gomoku, Connect6, Pente) and the Connect 4 rule variants (Pop Out, Pop 10,
Power Up).

It is load-bearing: every consumer takes an `engine=` factory, `Trajectory`
replays through the engine that recorded it, and the agent ladder searches with
:class:`agents.search.Position` rather than the drop primitives.

The claim is tested rather than asserted. `tests/free_placement.py` is a
free-placement engine — no gravity, a `rows * cols` action space, no compiled
rollout — and the conformance suite runs the full contract *and* the whole agent
ladder against it. At 3x3 it is tic-tac-toe, where `minimax:depth=9` searches
the game out and draws itself 40 times out of 40, which is the project's only
absolute check on playing strength.

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
