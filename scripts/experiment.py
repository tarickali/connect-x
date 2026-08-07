"""Run a config-driven experiment and append results to JSONL.

Experiments are described by a JSON file so a run is reproducible from a single
artifact — the spec plus the seed fully determine the numbers.

    python scripts/experiment.py experiments/ladder.json
    python scripts/experiment.py experiments/ladder.json --out results/ladder.jsonl

Spec format::

    {
      "name": "ladder",
      "agents": ["minimax:depth=4", "greedy"],
      "games": 200,
      "seed": 0,
      "workers": 4,
      "variants": {"shapes": [[6, 7], [8, 9]], "ks": [4], "player_counts": [2]}
    }

A ``"preset"`` or explicit ``"configs"`` list may be used instead of
``"variants"``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from connectx.config import config_id, make_config, preset  # noqa: E402
from connectx.variants import sweep, variant_grid  # noqa: E402


def configs_from_spec(spec: dict) -> list:
    if "configs" in spec:
        return [
            make_config(tuple(c["shape"]), c["k"], c["players"]) for c in spec["configs"]
        ]
    if "preset" in spec:
        return [preset(spec["preset"])]
    grid = spec.get("variants", {})
    return variant_grid(
        [tuple(s) for s in grid.get("shapes", [[6, 7]])],
        grid.get("ks", [4]),
        grid.get("player_counts", [2]),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("spec", type=Path, help="path to the experiment JSON")
    parser.add_argument("--out", type=Path, default=None, help="JSONL output path")
    args = parser.parse_args()

    spec = json.loads(args.spec.read_text())
    name = spec.get("name", args.spec.stem)
    configs = configs_from_spec(spec)
    if not configs:
        print("error: spec produced no valid variants", file=sys.stderr)
        return 2

    print(
        f"experiment '{name}': {len(configs)} variants x {spec.get('games', 100)} games"
    )
    result = sweep(
        spec["agents"],
        configs,
        games=spec.get("games", 100),
        seed=spec.get("seed", 0),
        workers=spec.get("workers", 1),
        on_variant=lambda m: print(f"  {config_id(m.config):<12} done", flush=True),
    )

    print()
    print(result.table())
    if result.skipped:
        print(f"\nskipped (seat count mismatch): {', '.join(result.skipped)}")

    out = args.out or _ROOT / "results" / f"{name}.jsonl"
    print(f"\nwrote {result.write_jsonl(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
