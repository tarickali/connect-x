# Usage

Every runnable command in the project, with what it is for.

- [Install](#install)
- [Specifying a variant](#specifying-a-variant)
- [Specifying an agent](#specifying-an-agent)
- [CLI commands](#cli-commands)
- [Recipes](#recipes)
- [Experiments](#experiments)
- [Library use](#library-use)
- [Training an agent](#training-an-agent)
- [Extending the project](#extending-the-project)
- [Development](#development)
- [Troubleshooting](#troubleshooting)

---

## Install

Requires Python 3.10+.

```bash
git clone https://github.com/tarickali/connect-x.git
cd connect-x

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -e .            # engine + agents + CLI
pip install -e ".[rl]"      # adds the Gymnasium / PettingZoo adapters
pip install -e ".[dev]"     # everything, plus pytest, ruff, and mypy
```

Verify:

```bash
connectx agents             # lists agents and presets
pytest                      # runs the test suite
```

> `numpy` is deliberately unpinned. `numba` declares its own compatible `numpy`
> range, so let pip resolve the pair. See [Troubleshooting](#troubleshooting).

---

## Specifying a variant

Every command that touches a game accepts the same four flags.

| Flag | Meaning | Example |
| --- | --- | --- |
| `--preset NAME` | A named variant | `--preset connect4` |
| `--shape ROWS COLS` | Board size | `--shape 8 9` |
| `--k N` | Tokens in a row to win | `--k 5` |
| `--players N` | Number of seats (tokens `1..N`) | `--players 3` |

Explicit flags override the preset, so `--preset connect4 --k 5` is a 6x7 board
that needs five in a row.

Presets: `tiny` (4x4 k=3) · `small` (5x6 k=4) · `connect4` (6x7 k=4) ·
`connect5` (9x10 k=5) · `wide` (6x12 k=4) · `tall` (12x6 k=4) ·
`three-player` (7x9 k=4) · `four-player` (8x10 k=4).

A variant must be playable: `1 <= k <= max(rows, cols)`, at least two distinct
player tokens in `1..255`. Anything else is rejected with an explanation rather
than producing a game nobody can win.

---

## Specifying an agent

Agents are named by a string, optionally with options: `name:key=value,key=value`.

| Agent | What it does | Useful options |
| --- | --- | --- |
| `random` | Uniform over legal columns | — |
| `greedy` | Take a win, block a loss, else play centre | — |
| `minimax` | Negamax + alpha-beta + transposition table (2 players) | `depth`, `time_limit`, `use_transpositions` |
| `mcts` | UCT with JIT rollouts (any number of players) | `simulations`, `time_limit`, `exploration`, `greedy_rollouts` |
| `human` | Reads columns from the keyboard | — |

```bash
minimax:depth=6
minimax:time_limit=0.5,depth=20          # iterative deepening under a time budget
mcts:simulations=2000,greedy_rollouts=false
```

The full spec string is what appears in result tables and JSON, so
`minimax:depth=4` and `minimax:depth=8` are distinct entrants.

---

## CLI commands

Installed as `connectx`; `python -m connectx` works identically without installing.

### `play` — watch one game

```bash
connectx play --preset connect4 --agents random random
connectx play --preset connect4 --agents human minimax:depth=6
connectx play --preset three-player --agents mcts greedy random
connectx play --preset connect4 --agents greedy random --quiet --seed 7
```

`--quiet` prints only the result. `--seed N` makes the game reproducible.

### `match` — many games, with confidence intervals

```bash
connectx match --preset connect4 --agents minimax:depth=4 greedy --games 200
connectx match --shape 8 9 --k 5 --agents mcts:simulations=800 greedy --games 100 --workers 8
connectx match --preset connect4 --agents greedy random --games 200 --json results/match.json
```

```
6x7k4p2  200 games  (1.09s, 3,620 moves/s)
  minimax:depth=4              wins  187  score  96.2%  95% CI [85.3%, 99.1%]
  greedy                       wins    0  score   3.8%  95% CI [ 0.9%, 14.7%]
  draws                                13
  seat wins: seat 0: 104, seat 1: 83
```

Seats are rotated so each entrant plays each position equally often — Connect
games are strongly first-player biased, and the `seat wins` line shows how much.
`--no-swap` disables it (not recommended). `--games` must be a multiple of the
player count while swapping is on.

`--workers N` runs games across processes. Seeds derive from the game index, so
the result is identical no matter how many workers you use.

### `tournament` — round robin with Elo

```bash
connectx tournament --preset connect4 \
  --agents random greedy mcts:simulations=400 minimax:depth=4 minimax:depth=6 \
  --games 100 --workers 8 --json results/ladder.json
```

```
agent                elo    score   games
-----------------------------------------
minimax:depth=5     2021   91.7%      90
minimax:depth=2     1831   73.3%      90
greedy              1378   35.0%      90
random               770    0.0%      90
```

Ratings come from a Bradley-Terry fit with a light prior, so an agent that wins
or loses everything still gets a finite number. Two-player variants only.

### `sweep` — one matchup across a grid of variants

This is the generalization measurement.

```bash
connectx sweep --agents minimax:depth=4 greedy \
  --shapes 5x6 6x7 7x9 9x10 --ks 3 4 5 --games 100 \
  --jsonl results/sweep.jsonl
```

```
minimax:depth=4 across 12 variants
variant     score            95% CI   draws    steps    moves/s
---------------------------------------------------------------
5x6k3p2    63.3%  [50.7%, 74.4%]       0      7.6      4,017
6x7k4p2    95.8%  [87.5%, 98.7%]       5     19.8      4,000
9x10k5p2  100.0%  [94.0%, 100.0%]      0     37.5      1,208
...
---------------------------------------------------------------
mean       93.4%   min 63.3%  max 100.0%  spread 36.7%
```

The `spread` is the headline: how much of the agent's edge survives a change of
board. Shapes accept `6x7` or `6,7`. Combinations where no win fits (`k=9` on a
4x5 board) are dropped automatically. `--player-counts 2 3 4` sweeps seat counts;
variants whose seat count does not match the number of agents are skipped and
listed.

### `surface` — round robin on every variant

The strongest form of the generalization measurement: instead of one matchup
per variant, run the whole field against itself on each board and compare the
ratings.

```bash
connectx surface --agents random greedy minimax:depth=4 mcts:simulations=400 \
  --shapes 5x6 6x7 9x10 12x12 --ks 3 4 5 --games 40 --workers 8 \
  --jsonl results/surface.jsonl
```

```
agent                      5x6k3p2    6x7k4p2   9x10k5p2  12x12k5p2
-------------------------------------------------------------------
minimax:depth=4              1,737      2,033      2,185      2,209
mcts:simulations=400         1,779      1,883      1,779      1,809
greedy                       1,623      1,369      1,345      1,316
random                         862        716        690        665
-------------------------------------------------------------------
spread                         917      1,317      1,495      1,544
rank of minimax:depth=4          2          1          1          1

ladder order is NOT stable across variants; moved: minimax:depth=4, mcts:simulations=400
```

Read it by column, never by row: each variant's ratings are fitted
independently and anchored to the same mean, so an agent's absolute Elo carries
no cross-variant meaning. What *does* transfer is the ordering, the gaps, and
the `spread` — a small spread means the variant fails to separate these agents
at all.

The last line is the headline: if the ladder reorders, a single-variant
benchmark would have given you the wrong ranking.

Two-player variants only, since a round robin is pairwise; anything else is
listed as skipped. Cost is `variants x pairings x games`, printed before the run
starts.

### `bench` — engine throughput

```bash
connectx bench --preset connect4
connectx bench --preset tiny --quick --json results/bench.json
```

Reports microseconds per primitive, `Game.transition` moves/second, batched
`VecGame` moves/second, and per-move agent cost. All timings exclude numba's
first-call compilation.

### `agents` — list what is available

```bash
connectx agents
```

---

## Recipes

Small, readable programs meant to be copied.

```bash
python recipes/class_example.py       # stateful Game API, RL-shaped step()
python recipes/functional_example.py  # pure JIT primitives, you hold the grid
python recipes/selfplay_example.py    # self-play -> ReplayMemory -> training tensors
```

`python main.py` runs one random-vs-random game; `python main.py <cli args>`
forwards to the CLI, so `python main.py match --help` works before installing.

---

## Experiments

Experiments are JSON files, so a run is reproducible from a single artifact.

```bash
python scripts/experiment.py experiments/ladder.json
python scripts/experiment.py experiments/ladder.json --out results/ladder.jsonl
```

```json
{
  "name": "ladder",
  "agents": ["minimax:depth=4", "greedy"],
  "games": 60,
  "seed": 0,
  "workers": 4,
  "variants": {
    "shapes": [[5, 6], [6, 7], [7, 9], [9, 10]],
    "ks": [3, 4, 5],
    "player_counts": [2]
  }
}
```

Use `"preset": "connect4"` for a single variant, or `"configs"` for an explicit
list. Results are written as JSONL — one record per variant, with win counts,
score rates, confidence intervals, seat splits, and throughput.

---

## Library use

### Class API

```python
from connectx import Game, preset
from agents import make_agent

config = preset("connect4")
agents = [make_agent("minimax:depth=6", config, seed=0),
          make_agent("greedy", config, seed=1)]

game = Game(config)
state, actions = game.start()
while not game.terminal():
    action = agents[state["info"]["active"]].select(state, actions)
    result = game.step(action)              # state, actions, rewards, terminated, report
    state, actions = result.state, result.actions

print(game.report())                        # {'winner': {...}, 'steps': 31, 'tie': False}
print(game.rewards())                       # array([ 1., -1.])
```

`transition(action)` is the game-shaped call returning `(state, actions)`;
`step(action)` is the RL-shaped one that also gives rewards and the terminal flag.

The grid returned by `state` is a **non-writeable view** — cheap to hand to an
agent, impossible to corrupt the game with. Call `np.array(grid)` for a mutable copy.

### Functional API

```python
import connectx.functional as cxf

grid = cxf.create_grid((6, 7))
if cxf.is_legal(grid, 3):                   # the functional layer does not raise
    grid = cxf.place_token(grid, 1, 3)
row = cxf.drop_row(grid, 3)
cxf.winner_at(grid, 4, row, 3)              # O(k) check through the last move
cxf.winner(grid, 4)                         # full scan, for arbitrary positions
```

### Search: undo and fork

```python
game = Game(config, undo=True)
game.start()
game.transition(3)
game.undo()                                 # cheap rewind, for tree search

branch = game.fork()                        # copy-on-write branch
branch.transition(0)                        # parent is untouched
```

### Batched environments

```python
import numpy as np
from connectx.vector import VecGame

vec = VecGame(preset("connect4"), num_envs=2048)
vec.reset()
rng = np.random.default_rng(0)
result = vec.step(vec.sample_actions(rng))
result.rewards        # (2048, 2)
result.terminated     # (2048,)
result.final_grids    # terminal boards, before autoreset cleared them
```

### Saving positions and datasets

```python
from connectx.utils import save, load
from connectx.dataset import save_trajectories, load_trajectories, build_supervised

save(game.instance(), "positions/opening.npz")
game2 = Game.from_instance(load("positions/opening.npz"))

save_trajectories("data/shard0.npz", trajectories)   # move lists, not boards
data = build_supervised(load_trajectories("data/shard0.npz"), mirror=True)
```

Both formats are `npz` + JSON, never `pickle`: loading a dataset should not be
able to execute code.

---

## Training an agent

### With PettingZoo (multi-agent, self-play)

```python
from connectx import preset
from connectx.adapters import make_aec_env

env = make_aec_env(preset("connect4"))
env.reset(seed=0)
for agent in env.agent_iter():
    observation, reward, terminated, truncated, info = env.last()
    if terminated or truncated:
        env.step(None)
        continue
    env.step(policy(observation["observation"], observation["action_mask"]))
```

### With Gymnasium (single agent vs a fixed opponent)

```python
from connectx.adapters import make_gym_env

env = make_gym_env(preset("connect4"), opponent="minimax:depth=4", seat=0)
observation, info = env.reset(seed=0)
observation, reward, terminated, truncated, info = env.step(action)
```

The opponent is an agent spec, so difficulty is a string in your config: start
at `"random"`, move to `"greedy"`, then `"minimax:depth=4"`.

### Observations

`encode_state` produces `(players + 1, rows, cols)` float32 planes. Plane 0 is
always the player to move, so one set of weights serves every seat, and board
size appears only as the spatial dimensions — a fully-convolutional model trained
on 6x7 can be evaluated on 8x9 unchanged.

```python
from connectx.encoding import encode_state, action_mask, mirror_action, observation_shape

observation_shape(config)          # (3, 6, 7)
planes = encode_state(state, config)
mask = action_mask(actions)        # bool, ready to add to logits
```

Mirror symmetry is exact for these games and doubles a dataset for free — pass
`mirror=True` to `build_supervised`, which reflects the policy targets to match.

---

## Extending the project

**A new agent.** Subclass `BaseAgent`, implement `select`, register it:

```python
from agents import REGISTRY
from agents.types import BaseAgent

class MyAgent(BaseAgent):
    def reset(self, config):        # called at the start of every episode
        super().reset(config)
    def select(self, state, actions):
        ...

REGISTRY["mine"] = MyAgent
```

Derive everything variant-specific in `reset(config)` rather than `__init__`, so
one instance can play any variant — that is what makes it measurable in a sweep.

**A new game.** Implement the `connectx.engine.GameEngine` protocol in a new
class — that is the intended route for pop moves, free placement, obstacles, or
different gravity.

Be aware the seam is not finished. Nothing consumes `GameEngine` yet: the arena,
both adapters, the benchmarks, and the CLI all construct `Game` directly, and
`Trajectory.replay` rebuilds positions with the gravity-based `place_token`, so
a free-placement game would replay incorrectly. Those call sites need routing
through an engine factory before a second game works end to end. See
[todo.md](todo.md) for the plan.

**A new preset.** Add it to `connectx.config.PRESETS`.

---

## Development

```bash
pytest                          # 292 tests
pytest -m "not rl"              # skip tests needing the [rl] extra
pytest --cov --cov-report=term-missing
ruff check connectx agents tests recipes scripts
ruff format connectx agents tests recipes scripts
mypy
```

CI runs the tests on Python 3.10-3.13, plus lint, types, and a smoke test that
executes every command on this page against a clean checkout.

### Is numba pulling its weight?

`scripts/ablation.py` answers that with measurements rather than opinion. It
compares the shipped JIT code against the same source uncompiled *and* against
an idiomatic numpy implementation:

```bash
python scripts/ablation.py                                # numba enabled
NUMBA_DISABLE_JIT=1 python scripts/ablation.py            # same code, no JIT
python scripts/ablation.py micro --json results/ablation.json
```

The whole test suite passes under `NUMBA_DISABLE_JIT=1`, so numba is a pure
accelerator here — never load-bearing for behaviour. Summary of the result is in
the [README](README.md#why-numba).

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'numba'`** — install the package rather
than only its imports: `pip install -e .`.

**`numba` and `numpy` version conflict.** `numba` supports a trailing range of
`numpy` versions. If pip installed a `numpy` that is too new, let it resolve
both together:

```bash
pip install --upgrade --force-reinstall "numba" "numpy"
```

Do not pin `numpy` yourself in `requirements.txt` — that is what causes the
mismatch.

**First call is slow.** numba compiles on first use. Compiled functions are
cached to `__pycache__`, so subsequent runs start fast; benchmarks warm up
before timing.

**`ModuleNotFoundError: No module named 'connectx'` when running a script** —
run from the repository root, or `pip install -e .`.

**Adapters raise ImportError** — `pip install -e ".[rl]"`.

**A match refuses to run**: `games must be a multiple of the player count` — seat
rotation needs balanced counts. Use a multiple of the seat count, or pass
`--no-swap` and accept the seat bias.
