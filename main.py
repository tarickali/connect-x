"""Entry point for `python main.py` — delegates to the connectx CLI.

Kept so a clean checkout has something obvious to run before installing.
`python main.py` with no arguments plays one random-vs-random game; anything
else is passed straight through, so `python main.py match --help` works too.

The runnable examples live in `recipes/`; see USAGE.md for every command.
"""

import sys

from connectx.cli import main

if __name__ == "__main__":
    argv = sys.argv[1:] or ["play", "--preset", "connect4", "--agents", "random", "random"]
    raise SystemExit(main(argv))
