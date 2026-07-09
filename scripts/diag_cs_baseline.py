"""Compatibility entry point for the C&S baseline diagnostic."""

from __future__ import annotations

import runpy
from pathlib import Path


TARGET = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "72fb8d9"
    / "scripts"
    / "diag_cs_baseline.py"
)


if __name__ == "__main__":
    runpy.run_path(str(TARGET), run_name="__main__")
