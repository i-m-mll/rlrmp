"""Compatibility entry point for the legacy Part 2.5 evaluator."""

from __future__ import annotations

import runpy
from pathlib import Path


TARGET = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "2ef67ca"
    / "scripts"
    / "eval_part2_5.py"
)


if __name__ == "__main__":
    runpy.run_path(str(TARGET), run_name="__main__")
