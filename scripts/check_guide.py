"""Verify that the numbers in the docs still match reality.

`read_guide.md` and `USAGE.md` quote test counts, coverage, and module sizes.
Those rot the moment anyone adds a file, and a review guide that lies about the
codebase is worse than no guide. This script re-derives each figure and diffs it
against what the docs claim.

    python scripts/check_guide.py           # report drift, exit 1 if any
    python scripts/check_guide.py --fix     # rewrite the docs with real numbers

It checks the *volatile* facts only — counts, sizes, versions. Prose about
design decisions is not machine-checkable and is not attempted.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GUIDE = ROOT / "read_guide.md"
USAGE = ROOT / "USAGE.md"


def source_lines(path: Path) -> int:
    """Non-blank lines, matching how the guide reports sizes."""
    return sum(1 for line in path.read_text().splitlines() if line.strip())


def run_pytest(*args: str) -> str:
    # No -q here: pyproject already sets addopts = "-q", and a second one means
    # -qq, which suppresses the summary line this script parses.
    done = subprocess.run(
        [sys.executable, "-m", "pytest", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return done.stdout


def passed_count(output: str) -> int:
    match = re.search(r"(\d+) passed", output)
    if match is None:
        raise RuntimeError(
            "could not read a pass count from pytest; the suite may be failing:\n"
            + output[-2000:]
        )
    return int(match.group(1))


def collect_facts() -> dict[str, int]:
    facts: dict[str, int] = {}

    facts["total_tests"] = passed_count(run_pytest())
    facts["conformance_total"] = passed_count(
        run_pytest("tests/test_engine_conformance.py")
    )
    facts["conformance_per_engine"] = passed_count(
        run_pytest("tests/test_engine_conformance.py", "-k", "Game/connect4")
    )
    facts["test_lines"] = sum(
        source_lines(p) for p in sorted((ROOT / "tests").glob("test_*.py"))
    )
    for path in sorted((ROOT / "connectx").rglob("*.py")) + sorted(
        (ROOT / "agents").glob("*.py")
    ):
        rel = path.relative_to(ROOT).as_posix()
        facts[f"lines:{rel}"] = source_lines(path)
    return facts


CHECKS = [
    # (label, fact key, regex with one capture group, files)
    ("total test count", "total_tests", r"\b(\d+) tests\b", [GUIDE, USAGE]),
    (
        "conformance total",
        "conformance_total",
        r"`tests/test_engine_conformance\.py` \| \*\*(\d+)\*\*",
        [GUIDE],
    ),
    (
        "conformance per engine",
        "conformance_per_engine",
        r"\*\*(\d+) checks against your engine\*\*",
        [USAGE],
    ),
    ("test line total", "test_lines", r"### 7\. Tests \(([\d,]+) lines", [GUIDE]),
]


def check(fix: bool) -> int:
    facts = collect_facts()
    problems: list[str] = []
    edits: dict[Path, str] = {}

    for label, key, pattern, files in CHECKS:
        expected = facts[key]
        for path in files:
            text = edits.get(path, path.read_text())
            found = re.findall(pattern, text)
            if not found:
                continue
            for raw in set(found):
                actual = int(raw.replace(",", ""))
                if actual == expected:
                    continue
                problems.append(f"{path.name}: {label} says {raw}, actually {expected:,}")
                if fix:
                    replacement = f"{expected:,}" if "," in raw else str(expected)
                    text = re.sub(
                        pattern,
                        lambda m, r=replacement: m.group(0).replace(m.group(1), r),
                        text,
                    )
                    edits[path] = text

    # Module sizes quoted in the guide's reading-order tables.
    guide_text = edits.get(GUIDE, GUIDE.read_text())
    for match in re.finditer(
        r"`((?:connectx|agents)/[\w/]+\.py)` \| (\d+) \|", guide_text
    ):
        rel, claimed = match.group(1), int(match.group(2))
        actual = facts.get(f"lines:{rel}")
        if actual is None:
            problems.append(f"read_guide.md: {rel} is listed but does not exist")
        elif actual != claimed:
            problems.append(
                f"read_guide.md: {rel} listed as {claimed} lines, actually {actual}"
            )
            if fix:
                guide_text = guide_text.replace(
                    f"`{rel}` | {claimed} |", f"`{rel}` | {actual} |"
                )
                edits[GUIDE] = guide_text

    if fix and edits:
        for path, text in edits.items():
            path.write_text(text)
        print(f"rewrote {', '.join(sorted(p.name for p in edits))}")
        return 0

    if problems:
        print("docs have drifted from the code:\n")
        for problem in problems:
            print(f"  {problem}")
        print("\nrun: python scripts/check_guide.py --fix")
        return 1

    print(f"docs match the code ({facts['total_tests']} tests, ", end="")
    print(f"{facts['conformance_per_engine']} conformance checks per engine)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--fix", action="store_true", help="rewrite the docs instead of reporting"
    )
    return check(parser.parse_args().fix)


if __name__ == "__main__":
    raise SystemExit(main())
