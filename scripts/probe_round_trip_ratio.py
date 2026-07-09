"""Compatibility entry point for the round-trip ratio diagnostic."""

from __future__ import annotations

import runpy
from pathlib import Path


TARGET = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "72fb8d9"
    / "scripts"
    / "probe_round_trip_ratio.py"
)


if __name__ == "__main__":
    runpy.run_path(str(TARGET), run_name="__main__")
