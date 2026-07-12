"""Error-only shim for the retired flag-per-field training interface."""

import sys


REPLACEMENT = (
    "PYTHONPATH=src uv run --no-sync python scripts/launch_training.py "
    "execute <authored-matrix.json>"
)


if __name__ == "__main__":
    print(REPLACEMENT, file=sys.stderr)
    raise SystemExit(2)
