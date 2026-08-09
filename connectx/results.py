"""Provenance and JSONL result files.

A result that cannot be traced back to the code and seed that produced it is
not a research result. Every record written by the harness therefore carries
enough context to reproduce it: the seed, the code version, and the numeric
stack it ran on.

Records are self-contained rather than sharing a file header, because JSONL is
useful precisely when you can `grep`, `sort`, and concatenate files without
tracking which header applies to which line.
"""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

__all__ = [
    "provenance",
    "git_revision",
    "append_jsonl",
    "read_jsonl",
    "completed_variants",
]

PathLike = str | Path


@lru_cache(maxsize=1)
def git_revision() -> dict[str, Any]:
    """Best-effort git SHA and dirty flag for the working tree.

    Returns ``{"sha": None, "dirty": None}`` outside a git checkout — an
    installed wheel is a legitimate way to run experiments, and that should not
    raise.
    """
    root = Path(__file__).resolve().parent.parent

    def run(*args: str) -> str | None:
        try:
            done = subprocess.run(
                ["git", *args],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return done.stdout.strip() if done.returncode == 0 else None

    sha = run("rev-parse", "HEAD")
    if sha is None:
        return {"sha": None, "dirty": None}
    status = run("status", "--porcelain")
    return {"sha": sha, "dirty": bool(status) if status is not None else None}


@lru_cache(maxsize=1)
def _environment() -> dict[str, Any]:
    import numpy

    from connectx import __version__

    try:
        import numba

        numba_version: str | None = numba.__version__
    except ImportError:  # pragma: no cover - numba is a hard dependency
        numba_version = None

    return {
        "connectx": __version__,
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "numba": numba_version,
        "platform": platform.platform(terse=True),
    }


def provenance(seed: int | None = None, **extra: Any) -> dict[str, Any]:
    """Everything needed to reproduce a result, minus the result itself.

    ``seed`` is the important field: the harness derives every per-game seed
    from it, so seed plus config plus code version fully determines the numbers.
    """
    record = {
        "seed": seed,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_revision(),
        "env": _environment(),
    }
    record.update(extra)
    return record


def append_jsonl(filepath: PathLike, record: dict[str, Any]) -> Path:
    """Append one record, creating parents as needed.

    Appending as each unit of work finishes is what makes a long run
    survivable: a sweep that dies at variant 11 of 12 keeps the first ten.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(record) + "\n")
    return path


def read_jsonl(filepath: PathLike) -> list[dict[str, Any]]:
    """Read a JSONL file, skipping blank lines. Missing files read as empty."""
    path = Path(filepath)
    if not path.exists():
        return []
    records = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def completed_variants(filepath: PathLike) -> set[str]:
    """Variant ids already present in a result file, for resuming a run."""
    return {
        str(record["variant"]) for record in read_jsonl(filepath) if "variant" in record
    }
